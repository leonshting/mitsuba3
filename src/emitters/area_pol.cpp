#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>
#include <drjit/traversable_base.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-area:

Area light (:monosp:`area`)
---------------------------

.. pluginparameters::

 * - radiance
   - |spectrum| or |texture|
   - Specifies the emitted radiance in units of power per unit area per unit steradian.
   - |exposed|, |differentiable|

This plugin implements an area light, i.e. a light source that emits
diffuse illumination from the exterior of an arbitrary shape.
Since the emission profile of an area light is completely diffuse, it
has the same apparent brightness regardless of the observer's viewing
direction. Furthermore, since it occupies a nonzero amount of space, an
area light generally causes scene objects to cast soft shadows.

To create an area light source, simply instantiate the desired
emitter shape and specify an :monosp:`area` instance as its child:

.. tabs::
    .. code-tab:: xml
        :name: sphere-light

        <shape type="sphere">
            <emitter type="area">
                <rgb name="radiance" value="1.0"/>
            </emitter>
        </shape>

    .. code-tab:: python

        'type': 'sphere',
        'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'rgb',
                'value': 1.0,
            }
        }

 */

template <typename Float, typename Spectrum>
class AreaPolLight final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_shape, m_medium)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    AreaPolLight(const Properties &props) : Base(props) {
        if (props.has_property("to_world"))
            Throw("Found a 'to_world' transformation -- this is not allowed. "
                  "The area light inherits this transformation from its parent "
                  "shape.");
        m_radiance = props.get_emissive_texture<Texture>("radiance", 1.f);

        m_flags = +EmitterFlags::Surface;
        if (m_radiance->is_spatially_varying())
            m_flags |= +EmitterFlags::SpatiallyVarying;
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("radiance", m_radiance, ParamFlags::Differentiable);
    }

    AffineTransform4f world_transform() const override {
        return this->m_shape->world_transform();
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        
        // si.wi is inverse of outgoing direction
        // TODO: leonsht
        // need to align stokes vector in form of muller matrix into 
        // align frame with incoming ray dir
        auto depolarized = depolarizer<Spectrum>(m_radiance->eval(si, active));    
        if constexpr (!is_polarized_v<Spectrum>) {
            return depolarized;
        } else {
            Spectrum tx_ay_mueller = mueller::linear_polarizer(1.f);
            
            Vector3f forward_world = si.to_world(si.wi);
            Vector3f forward_local = this->world_transform().inverse() * forward_world;

            Vector3f a_axis_local = Vector3f(0.f, 1.f, 0.f);
            Vector3f a_axis_world = this->world_transform() * a_axis_local;
            
            Vector3f eff_a_axis_world = dr::normalize(
                a_axis_world - dr::dot(a_axis_world, forward_world) * forward_world
            );
            Vector3f eff_t_axis_world = dr::cross(forward_world, eff_a_axis_world);
            // Vector3f eff_t_axis_local = this->world_transform().inverse() * eff_t_axis_world;
            Vector3f stokes_basis_world = mueller::stokes_basis(forward_world);

            Spectrum M = mueller::rotate_mueller_basis_collinear(
                tx_ay_mueller, forward_world, eff_t_axis_world, stokes_basis_world
            );

            return M * depolarized;
        }
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2, const Point2f &sample3,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample spatial component
        auto [ps, pos_weight] = sample_position(time, sample2, active);

        // 2. Sample directional component
        Vector3f local = warp::square_to_cosine_hemisphere(sample3);

        // 3. Sample spectral component
        SurfaceInteraction3f si(ps, dr::zeros<Wavelength>());
        auto [wavelength, wav_weight] =
            sample_wavelengths(si, wavelength_sample, active);
        si.time = time;
        si.wavelengths = wavelength;

        // Note: some terms cancelled out with `warp::square_to_cosine_hemisphere_pdf`.
        Spectrum weight = pos_weight * wav_weight * dr::Pi<ScalarFloat>;
        auto depolarized_weight = depolarizer<Spectrum>(weight);

        if constexpr (!is_polarized_v<Spectrum>) {
            return { si.spawn_ray(si.to_world(local)), depolarized_weight };
        } else {
            Spectrum tx_ay_mueller = mueller::linear_polarizer(1.f);
            
            Vector3f forward_world = si.to_world(local);
            Vector3f a_axis_local = Vector3f(0.f, 1.f, 0.f);
            Vector3f forward_local = this->world_transform().inverse() * forward_world;
            
            Vector3f eff_a_axis_local = dr::normalize(
                a_axis_local - dr::dot(a_axis_local, forward_local) * forward_local
            );
            Vector3f eff_t_axis_local = dr::cross(forward_local, eff_a_axis_local);
            Vector3f stokes_basis_local = mueller::stokes_basis(forward_local);

            Spectrum M = mueller::rotate_mueller_basis_collinear(
                tx_ay_mueller, forward_local, eff_t_axis_local, stokes_basis_local
            );

            return { si.spawn_ray(si.to_world(local)), M * depolarized_weight };
        }
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        if constexpr (drjit::is_jit_v<Float>) {
            if (!m_shape)
                return { dr::zeros<DirectionSample3f>(), 0.f };
        } else {
            Assert(m_shape, "Can't sample from an area emitter without an "
                            "associated Shape.");
        }

        DirectionSample3f ds;
        SurfaceInteraction3f si;

        // One of two very different strategies is used depending on 'm_radiance'
        if (likely(!m_radiance->is_spatially_varying())) {
            // Texture is uniform, try to importance sample the shape wrt. solid angle at 'it'
            ds = m_shape->sample_direction(it, sample, active);
            active &= dr::dot(ds.d, ds.n) < 0.f && (ds.pdf != 0.f);

            si = SurfaceInteraction3f(ds, it.wavelengths);
        } else {
            // Importance sample the texture, then map onto the shape
            auto [uv, pdf] = m_radiance->sample_position(sample, active);
            active &= (pdf != 0.f);

            si = m_shape->eval_parameterization(uv, +RayFlags::All, active);
            si.wavelengths = it.wavelengths;
            active &= si.is_valid();

            ds.p = si.p;
            ds.n = si.n;
            ds.uv = si.uv;
            ds.time = it.time;
            ds.delta = false;
            ds.d = ds.p - it.p;

            Float dist_squared = dr::squared_norm(ds.d);
            ds.dist = dr::sqrt(dist_squared);
            ds.d /= ds.dist;

            Float dp = dr::dot(ds.d, ds.n);
            active &= dp < 0.f;
            ds.pdf = dr::select(active, pdf / dr::norm(dr::cross(si.dp_du, si.dp_dv)) *
                                        dist_squared / -dp, 0.f);
        }

        UnpolarizedSpectrum spec = m_radiance->eval(si, active) / ds.pdf;
        ds.emitter = this;

        // TODO: leonshtalign polarization with outgoing with -si.d ?? 
        // wi + local shading frame = love

        auto depolarized = depolarizer<Spectrum>(spec) & active;
        if constexpr (!is_polarized_v<Spectrum>) {
            return { ds, depolarized };
        } else {
            Spectrum tx_ay_mueller = mueller::linear_polarizer(1.f);
            
            Vector3f forward_world = ds.d;
            Vector3f a_axis_local = Vector3f(0.f, 1.f, 0.f);
            Vector3f forward_local = this->world_transform().inverse() * forward_world;
            
            Vector3f eff_a_axis_local = dr::normalize(
                a_axis_local - dr::dot(a_axis_local, forward_local) * forward_local
            );
            Vector3f eff_t_axis_local = dr::cross(forward_local, eff_a_axis_local);
            Vector3f stokes_basis_local = mueller::stokes_basis(forward_local);

            Spectrum M = mueller::rotate_mueller_basis_collinear(
                tx_ay_mueller, forward_local, eff_t_axis_local, stokes_basis_local
            );
            return { ds, M * depolarized };
        }

    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        Float dp = dr::dot(ds.d, ds.n);
        active &= dp < 0.f;

        if constexpr (drjit::is_jit_v<Float>) {
            if (!m_shape)
                return 0.f;
        } else {
            Assert(m_shape,
                   "The area emitter without has no associated Shape!");
        }

        Float value;
        if (!m_radiance->is_spatially_varying()) {
            value = m_shape->pdf_direction(it, ds, active);
        } else {
            // This surface intersection would be nice to avoid..
            SurfaceInteraction3f si = m_shape->eval_parameterization(ds.uv, +RayFlags::dPdUV, active);
            active &= si.is_valid();

            value = m_radiance->pdf_position(ds.uv, active) * dr::square(ds.dist) /
                    (dr::norm(dr::cross(si.dp_du, si.dp_dv)) * -dp);
        }

        return dr::select(active, value, 0.f);
    }

    Spectrum eval_direction(const Interaction3f &it,
                            const DirectionSample3f &ds,
                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        Float dp = dr::dot(ds.d, ds.n);
        active &= dp < 0.f;

        SurfaceInteraction3f si(ds, it.wavelengths);
        UnpolarizedSpectrum spec = m_radiance->eval(si, active);

        auto depolarized = dr::select(active, depolarizer<Spectrum>(spec), 0.f);
        if constexpr (!is_polarized_v<Spectrum>) {
            return dr::select(active, depolarized, 0.f);
        } else {
            Spectrum tx_ay_mueller = mueller::linear_polarizer(1.f);
            
            Vector3f forward_world = ds.d;
            Vector3f a_axis_local = Vector3f(0.f, 1.f, 0.f);
            Vector3f forward_local = this->world_transform().inverse() * forward_world;
            
            Vector3f eff_a_axis_local = dr::normalize(
                a_axis_local - dr::dot(a_axis_local, forward_local) * forward_local
            );
            Vector3f eff_t_axis_local = dr::cross(forward_local, eff_a_axis_local);
            Vector3f stokes_basis_local = mueller::stokes_basis(forward_local);

            Spectrum M = mueller::rotate_mueller_basis_collinear(
                tx_ay_mueller, forward_local, eff_t_axis_local, stokes_basis_local
            );

            return M * depolarized;
        }
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float time, const Point2f &sample,
                    Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSamplePosition, active);

        if constexpr (drjit::is_jit_v<Float>) {
            if (!m_shape)
                return { dr::zeros<PositionSample3f>(), 0.f };
        } else {
            Assert(m_shape, "Cannot sample from an area emitter without an "
                            "associated Shape.");
        }

        // Two strategies to sample the spatial component based on 'm_radiance'
        PositionSample3f ps;
        if (!m_radiance->is_spatially_varying()) {
            // Radiance not spatially varying, use area-based sampling of shape
            ps = m_shape->sample_position(time, sample, active);
        } else {
            // Importance sample texture
            auto [uv, pdf] = m_radiance->sample_position(sample, active);
            active &= (pdf != 0.f);

            auto si = m_shape->eval_parameterization(uv, +RayFlags::All, active);
            active &= si.is_valid();
            pdf /= dr::norm(dr::cross(si.dp_du, si.dp_dv));

            ps = si;
            ps.pdf = pdf;
            ps.delta = false;
        }

        Float weight = dr::select(active && ps.pdf > 0.f, dr::rcp(ps.pdf), Float(0.f));
        return { ps, weight };
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        return m_radiance->sample_spectrum(
            si, math::sample_shifted<Wavelength>(sample), active);
    }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "AreaPolLight[" << std::endl
            << "  radiance = " << string::indent(m_radiance) << "," << std::endl
            << "  surface_area = ";
        if (m_shape) oss << m_shape->surface_area();
        else         oss << "  <no shape attached!>";
        oss << "," << std::endl;
        if (m_medium) oss << string::indent(m_medium);
        else         oss << "  <no medium attached!>";
        oss << std::endl << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(AreaPolLight)
private:
    ref<Texture> m_radiance;

    MI_TRAVERSE_CB(Base, m_radiance)
};

MI_EXPORT_PLUGIN(AreaPolLight)
NAMESPACE_END(mitsuba)
