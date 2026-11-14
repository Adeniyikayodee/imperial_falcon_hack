// Wind at a point: the browser mirror of falcon.live.field (surface layer, gust fronts, site gain).
// One copy for every page; tests/test_field_js.py holds it to the Python to 1e-12 m/s.
function falconWind(F, x, y, z, t, out) {
  const z0 = F.site.z0, sp = Math.hypot(F.mean[0], F.mean[1]), ux = sp > 0.2 ? F.mean[0] / sp : 1, uy = sp > 0.2 ? F.mean[1] / sp : 0;
  const prof = Math.log(Math.max(z, 2 * z0) / z0) / Math.log(10 / z0);
  out[0] = F.mean[0] * prof; out[1] = F.mean[1] * prof; out[2] = 0;
  if (F.gust > 0 && sp > 0) { // von Karman turbulence frozen into the air, advected at the local mean speed
    const xi = x * ux + y * uy - Math.max(sp * prof, 1.5) * t, base = F.gust * F.kappa * sp / Math.log(10 / z0), c = [0, 0, 0];
    for (let j = 0; j < 3; j++) for (let i = 0; i < F.k.length; i++) c[j] += F.a[j][i] * Math.sin(F.k[i] * xi + F.phase[j][i]);
    const su = base * F.sigma[0] * c[0], sv = base * F.sigma[1] * c[1];
    out[0] += su * ux - sv * uy; out[1] += su * uy + sv * ux; out[2] += base * F.sigma[2] * c[2];
  }
  for (const b of F.bursts) { // sin^2 gust fronts sweeping in from upwind
    const tau = (t - b.t0) - (b.dir[0] * x + b.dir[1] * y + b.dir[2] * z + F.burst_from) / F.burst_v;
    if (tau >= 0 && tau <= F.burst_s) { const s = Math.sin(Math.PI * tau / F.burst_s) ** 2 * b.speed; for (let j = 0; j < 3; j++) out[j] += s * b.dir[j]; }
  }
  const [ga, gc] = falconSiteGain(F.site, x, z); out[0] *= ga; out[1] *= gc;
  return out;
}
// street canyon gain (along x, across y): channelled along, sheltered across, jetting through cross-street gaps
function falconSiteGain(S, x, z) {
  if (!S.city) return [1, 1];
  const below = S.city * Math.min(1, Math.max(0, (S.roof - z) / 2));
  const gap = Math.min(1, S.gaps.reduce((a, g) => a + Math.exp(-(((x - g) / S.gap_w) ** 2)), 0));
  return [1 + below * (S.channel - 1), 1 + below * (S.shelter + (S.jet - S.shelter) * gap - 1)];
}
if (typeof module !== "undefined") module.exports = { falconWind, falconSiteGain };
