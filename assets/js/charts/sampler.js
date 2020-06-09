// Original Trace to select (heatmap)
import {heatmap} from "/assets/js/charts/heatmap.js";
import {grid} from "/assets/js/charts/grid.js";


Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};

let xdim = math.sqrt(grid.size()[0]);

Plotly.newPlot('graph', heatmap.data, heatmap.layout, {displayModeBar: false});

var myPlot = document.getElementById('graph');
myPlot.on("plotly_click", function (data) {
  let points = data.points[0]
  if ((points.curveNumber == 3) || (points.curveNumber == 4)) {
    console.log("Selected heatmap!");
    let weights = [-1 * points.x, -1* points.y];
    let z = math.matrix(math.exp(math.multiply(grid, weights)));
    z = z.map(function (value, index, matrix) {
      return math.divide(1.0 , 1 + value).clamp(0.001, 0.999);
    });

    heatmap.data[2].z = math.reshape(z, [xdim, xdim]).valueOf();
    heatmap.data[4].x = [points.x];
    heatmap.data[4].y = [points.y];

    Plotly.react('graph', heatmap.data, heatmap.layout, {displayModeBar: false});
  }
});