"""The pages draw air and sway trees with web/field.js; the drones fly through falcon.live.field.
If the two drift, the screen shows wind the drones never felt, so hold them equal."""
import json
import shutil
import subprocess
from pathlib import Path

import jax
import numpy as np
import pytest

import falcon  # noqa: F401
from falcon.live import Live, field

JS = Path(__file__).resolve().parent.parent / "web" / "field.js"
RUN = f"const {{falconWind}} = require({json.dumps(str(JS))}); const d = JSON.parse(require('fs').readFileSync(0));" \
      "console.log(JSON.stringify(d.pts.map((p) => falconWind(d.F, p[0], p[1], p[2], p[3], [0, 0, 0]))));"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("site", ["open", "city"])
def test_page_wind_matches_the_simulator(site):
    live = Live(traj="street", site=site)
    live.set_wind(mean=[2.0, 3.5], gust=1.0)
    live.k = 2500
    live.burst(2.0, (0.6, 0.8))
    pts = np.asarray(jax.random.uniform(jax.random.PRNGKey(3), (200, 4), minval=-1.0) * np.array([20, 8, 10, 3])
                     + np.array([0, 0, 12, live.t + 1.5]))
    py = np.stack([field(p[:3], p[3], live.mean, live.turb, live.gust, live.b_t0, live.b_dir, live.b_speed, live.site)
                   for p in pts])
    out = subprocess.run(["node", "-e", RUN], input=json.dumps({"F": live.field_spec(), "pts": pts.tolist()}),
                         capture_output=True, text=True, check=True)
    np.testing.assert_allclose(np.array(json.loads(out.stdout)), py, atol=1e-12)
