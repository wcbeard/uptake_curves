import json
from pathlib import Path

html_template = """
<!DOCTYPE html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@4"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head>

<body>
  <div id="win"></div>
  <div id="mac"></div>
  <div id="linux"></div>

  <script>
    const win_spec = {win_spec};
    const mac_spec = {mac_spec};
    const linux_spec = {linux_spec};

    vegaEmbed("#win", win_spec)
        // result.view provides access to the Vega View API
      .then(result => console.log(result))
      .catch(console.warn);

    vegaEmbed("#mac", mac_spec)
        // result.view provides access to the Vega View API
      .then(result => console.log(result))
      .catch(console.warn);

    vegaEmbed("#linux", linux_spec)
        // result.view provides access to the Vega View API
      .then(result => console.log(result))
      .catch(console.warn);

  </script>
</body>
"""

json_base = Path("/Users/wbeard/repos/uptake_curves/reports/channel_html")


def render_channel(win, mac, linux, channel):
    html = html_template.format(
        win_spec=json.dumps(win.to_dict()),
        mac_spec=json.dumps(mac.to_dict()),
        linux_spec=json.dumps(linux.to_dict()),
    )
    out = json_base / f"{channel}.html"
    with open(out, "w") as fp:
        fp.write(html)
