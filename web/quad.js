// Shared scene pieces for the demo pages: canvas textures, compass names and the Crazyflie-style
// micro quadcopter (real dimensions times S, frame tinted in the drone's colour, ghost drawn translucent).
const compass = (d) => ["west", "south-west", "south", "south-east", "east", "north-east", "north", "north-west"][((Math.round(d / 45) % 8) + 8) % 8];
function tex(w, h, paint, repeat) {
  const c = document.createElement("canvas"); c.width = w; c.height = h; paint(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8; t.encoding = THREE.sRGBEncoding; if (repeat) t.repeat.set(...repeat); return t;
}
const BLUR = tex(128, 128, (g) => { const r = g.createRadialGradient(64, 64, 6, 64, 64, 62); r.addColorStop(0, "rgba(30,30,30,0)"); r.addColorStop(0.25, "rgba(30,30,30,.35)"); r.addColorStop(0.85, "rgba(30,30,30,.22)"); r.addColorStop(1, "rgba(30,30,30,0)"); g.fillStyle = r; g.beginPath(); g.arc(64, 64, 62, 0, 7); g.fill(); });
BLUR.encoding = THREE.LinearEncoding;
function makeDrone(accent, ghost, S) {
  const g = new THREE.Group(), a = 0.0325 * S; // motor offset along body x and y, as in falcon/params.py
  const mat = (o) => new THREE.MeshStandardMaterial(Object.assign(o, ghost ? { transparent: true, opacity: 0.7, depthWrite: false } : {}));
  // frame tinted in the drone's own colour so the two read apart at a glance
  const pcb = mat({ color: 0x1b1f22, roughness: 0.5, metalness: 0.3 }), gold = mat({ color: 0xc9a44c, roughness: 0.35, metalness: 0.9 });
  const white = mat({ color: accent, roughness: 0.5, emissive: accent, emissiveIntensity: 0.35 }), metal = mat({ color: 0xb8bcc2, roughness: 0.25, metalness: 0.95 });
  const add = (geo, m, x, y, z) => { const o = new THREE.Mesh(geo, m); o.position.set(x, y, z); o.castShadow = !ghost; g.add(o); return o; };
  add(new THREE.BoxGeometry(0.036 * S, 0.036 * S, 0.0016 * S), pcb, 0, 0, 0);
  add(new THREE.BoxGeometry(0.008 * S, 0.008 * S, 0.0012 * S), gold, 0.006 * S, 0.004 * S, 0.0014 * S);
  add(new THREE.BoxGeometry(0.03 * S, 0.0025 * S, 0.004 * S), pcb, 0, 0.012 * S, 0.003 * S);
  add(new THREE.BoxGeometry(0.03 * S, 0.0025 * S, 0.004 * S), pcb, 0, -0.012 * S, 0.003 * S);
  add(new THREE.BoxGeometry(0.03 * S, 0.017 * S, 0.0075 * S), mat({ color: accent, roughness: 0.45, metalness: 0.2, emissive: accent, emissiveIntensity: 0.1 }), 0, 0, -0.0048 * S);
  const props = [], spin = [-1, 1, -1, 1];
  [[1, -1], [-1, -1], [-1, 1], [1, 1]].forEach(([sx, sy], i) => {
    const arm = add(new THREE.BoxGeometry(Math.hypot(a, a) - 0.018 * S, 0.004 * S, 0.0025 * S), white, sx * a * 0.55, sy * a * 0.55, 0); arm.rotation.z = Math.atan2(sy, sx);
    const leg = add(new THREE.CylinderGeometry(0.0012 * S, 0.0008 * S, 0.012 * S, 8), white, sx * a, sy * a, -0.007 * S); leg.rotation.x = Math.PI / 2;
    const motor = add(new THREE.CylinderGeometry(0.0035 * S, 0.0035 * S, 0.012 * S, 16), metal, sx * a, sy * a, 0.002 * S); motor.rotation.x = Math.PI / 2;
    const hub = new THREE.Group(); hub.position.set(sx * a, sy * a, 0.0095 * S); g.add(hub);
    const bm = mat({ color: i % 2 ? 0xf4f4f4 : 0x202326, roughness: 0.4 });
    for (const k of [1, -1]) { const b = new THREE.Mesh(new THREE.BoxGeometry(0.0215 * S, 0.0042 * S, 0.0006 * S), bm); b.position.x = k * 0.0115 * S; b.rotation.x = k * 0.25; hub.add(b); }
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.0235 * S, 40), new THREE.MeshBasicMaterial({ map: BLUR, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    disc.position.copy(hub.position); disc.position.z += 0.0003 * S; g.add(disc);
    props.push({ hub, disc, dir: spin[i] });
  });
  if (!ghost) {
    const led = (c, x, y) => add(new THREE.SphereGeometry(0.0018 * S, 10, 8), new THREE.MeshBasicMaterial({ color: c }), x, y, -0.0012 * S);
    led(0xff2a2a, 0.016 * S, 0.016 * S); led(0x2aff6a, 0.016 * S, -0.016 * S); led(0x3a7bff, -0.016 * S, 0.016 * S); led(0x3a7bff, -0.016 * S, -0.016 * S);
  }
  return { g, props };
}
