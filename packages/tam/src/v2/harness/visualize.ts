/**
 * Generate HTML visualization for TAM v2 training runs
 *
 * Usage:
 *   bun src/v2/harness/visualize.ts <path-to-run-directory>
 *   bun src/v2/harness/visualize.ts runs/1d-damped-spring-1767552895223
 */

import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

interface CheckpointData {
  sample: number;
  portCount: number;
  // Test metrics
  testAgency: number;
  testError: number;
  testBindingRate: number;
  // Training metrics
  trainAgency: number;
  trainError: number;
  trainBindingRate: number;
  // Calibration metrics
  testCalibrationGap?: number;
  trainCalibrationGap?: number;
}

async function generateVisualization(runDir: string) {
  // Read checkpoints data
  const checkpointsPath = join(runDir, "checkpoints.jsonl");
  const checkpointsText = await readFile(checkpointsPath, "utf-8");
  const data: CheckpointData[] = checkpointsText
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));

  // Read metadata
  const metadataPath = join(runDir, "metadata.json");
  const metadata = JSON.parse(await readFile(metadataPath, "utf-8"));

  // Generate HTML with embedded data
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${metadata.experiment} - TAM v2 Training</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 20px;
      background: #f5f5f5;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
      background: white;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    h1 {
      margin-top: 0;
      color: #333;
    }

    .metadata {
      background: #f8f8f8;
      padding: 15px;
      border-radius: 4px;
      margin-bottom: 20px;
      font-size: 14px;
    }

    .metadata code {
      background: #e0e0e0;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 12px;
    }

    .chart {
      margin-top: 20px;
    }

    .line {
      fill: none;
      stroke-width: 2.5;
    }

    .line-error-test {
      stroke: #e74c3c;
    }

    .line-error-train {
      stroke: #e74c3c;
      stroke-dasharray: 5,5;
      opacity: 0.7;
    }

    .line-agency-test {
      stroke: #3498db;
    }

    .line-agency-train {
      stroke: #3498db;
      stroke-dasharray: 5,5;
      opacity: 0.7;
    }

    .line-binding-test {
      stroke: #2ecc71;
    }

    .line-binding-train {
      stroke: #2ecc71;
      stroke-dasharray: 5,5;
      opacity: 0.7;
    }

    .line-calibration-test {
      stroke: #9b59b6;
    }

    .line-calibration-train {
      stroke: #9b59b6;
      stroke-dasharray: 5,5;
      opacity: 0.7;
    }

    .axis-label {
      font-size: 12px;
      font-weight: 600;
    }

    .legend {
      font-size: 14px;
    }

    .grid line {
      stroke: #e0e0e0;
      stroke-opacity: 0.7;
      shape-rendering: crispEdges;
    }

    .grid path {
      stroke-width: 0;
    }

    .tooltip {
      position: absolute;
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 12px;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.2s;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>${metadata.experiment}</h1>

    <div class="metadata">
      <strong>Run ID:</strong> <code>${metadata.runId}</code><br>
      <strong>Timestamp:</strong> ${new Date(metadata.timestamp).toLocaleString()}<br>
      <strong>Final Ports:</strong> ${data[data.length - 1]!.portCount}
    </div>

    <div id="chart" class="chart"></div>
    <div id="tooltip" class="tooltip"></div>
  </div>

  <script>
    const data = ${JSON.stringify(data, null, 2)};

    // Chart dimensions
    const margin = { top: 20, right: 60, bottom: 20, left: 60 };
    const width = 1100 - margin.left - margin.right;
    const chartHeight = 180;
    const spacing = 40;

    // Create SVG container (4 charts now)
    const totalHeight = (chartHeight * 4) + (spacing * 3) + margin.top + margin.bottom;
    const svg = d3.select('#chart')
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', totalHeight);

    // Shared X scale
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.sample)])
      .range([0, width]);

    // Y scales for each metric
    const maxError = Math.max(
      d3.max(data, d => d.testError),
      d3.max(data, d => d.trainError)
    );
    const yScaleError = d3.scaleLinear()
      .domain([0, maxError * 1.1])
      .range([chartHeight, 0]);

    const yScaleAgency = d3.scaleLinear()
      .domain([0, 1])
      .range([chartHeight, 0]);

    const yScaleBinding = d3.scaleLinear()
      .domain([0, 1])
      .range([chartHeight, 0]);

    // Y scale for calibration gap
    const maxCalGap = Math.max(
      d3.max(data, d => d.testCalibrationGap || 0),
      d3.max(data, d => d.trainCalibrationGap || 0),
      0.1 // minimum scale
    );
    const yScaleCalibration = d3.scaleLinear()
      .domain([0, maxCalGap * 1.1])
      .range([chartHeight, 0]);

    // Helper function to create a chart with train/test lines
    function createDualChart(yOffset, yScale, baseColor, baseName, yLabel, getTestValue, getTrainValue) {
      const g = svg.append('g')
        .attr('transform', \`translate(\${margin.left},\${yOffset})\`);

      // Grid
      g.append('g')
        .attr('class', 'grid')
        .attr('transform', \`translate(0,\${chartHeight})\`)
        .call(d3.axisBottom(xScale).tickSize(-chartHeight).tickFormat(''));

      g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(yScale).tickSize(-width).tickFormat(''));

      // Test line (solid)
      const testLine = d3.line()
        .x(d => xScale(d.sample))
        .y(d => yScale(getTestValue(d)))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(data)
        .attr('class', \`line line-\${baseName}-test\`)
        .attr('d', testLine);

      // Train line (dashed)
      const trainLine = d3.line()
        .x(d => xScale(d.sample))
        .y(d => yScale(getTrainValue(d)))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(data)
        .attr('class', \`line line-\${baseName}-train\`)
        .attr('d', trainLine);

      // Y axis
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -chartHeight / 2)
        .attr('y', -45)
        .attr('fill', baseColor)
        .attr('text-anchor', 'middle')
        .text(yLabel);

      // Legend
      const legend = g.append('g')
        .attr('transform', \`translate(\${width - 100}, 10)\`);

      legend.append('line')
        .attr('x1', 0).attr('x2', 20)
        .attr('y1', 0).attr('y2', 0)
        .attr('stroke', baseColor)
        .attr('stroke-width', 2);
      legend.append('text')
        .attr('x', 25).attr('y', 4)
        .style('font-size', '11px')
        .text('Test');

      legend.append('line')
        .attr('x1', 0).attr('x2', 20)
        .attr('y1', 15).attr('y2', 15)
        .attr('stroke', baseColor)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .attr('opacity', 0.7);
      legend.append('text')
        .attr('x', 25).attr('y', 19)
        .style('font-size', '11px')
        .text('Train');

      return g;
    }

    // Tooltip
    const tooltip = d3.select('#tooltip');

    // Chart 1: Error
    const errorChart = createDualChart(
      margin.top,
      yScaleError,
      '#e74c3c',
      'error',
      'Error',
      d => d.testError,
      d => d.trainError
    );

    errorChart.append('rect')
      .attr('width', width)
      .attr('height', chartHeight)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .on('mousemove', function(event) {
        const [mouseX] = d3.pointer(event);
        const sample = Math.round(xScale.invert(mouseX));
        const dataPoint = data.find(d => d.sample >= sample);
        if (dataPoint) {
          tooltip
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(\`<strong>Sample \${dataPoint.sample}</strong><br/>Test Error: \${dataPoint.testError.toFixed(4)}<br/>Train Error: \${dataPoint.trainError.toFixed(4)}\`);
        }
      })
      .on('mouseout', () => tooltip.style('opacity', 0));

    // Chart 2: Agency
    const agencyChart = createDualChart(
      margin.top + chartHeight + spacing,
      yScaleAgency,
      '#3498db',
      'agency',
      'Agency',
      d => d.testAgency,
      d => d.trainAgency
    );

    agencyChart.append('rect')
      .attr('width', width)
      .attr('height', chartHeight)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .on('mousemove', function(event) {
        const [mouseX] = d3.pointer(event);
        const sample = Math.round(xScale.invert(mouseX));
        const dataPoint = data.find(d => d.sample >= sample);
        if (dataPoint) {
          tooltip
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(\`<strong>Sample \${dataPoint.sample}</strong><br/>Test Agency: \${(dataPoint.testAgency * 100).toFixed(1)}%<br/>Train Agency: \${(dataPoint.trainAgency * 100).toFixed(1)}%\`);
        }
      })
      .on('mouseout', () => tooltip.style('opacity', 0));

    // Chart 3: Binding Rate
    const bindingChart = createDualChart(
      margin.top + (chartHeight + spacing) * 2,
      yScaleBinding,
      '#2ecc71',
      'binding',
      'Binding Rate',
      d => d.testBindingRate,
      d => d.trainBindingRate
    );

    bindingChart.append('rect')
      .attr('width', width)
      .attr('height', chartHeight)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .on('mousemove', function(event) {
        const [mouseX] = d3.pointer(event);
        const sample = Math.round(xScale.invert(mouseX));
        const dataPoint = data.find(d => d.sample >= sample);
        if (dataPoint) {
          tooltip
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(\`<strong>Sample \${dataPoint.sample}</strong><br/>Test Binding: \${(dataPoint.testBindingRate * 100).toFixed(1)}%<br/>Train Binding: \${(dataPoint.trainBindingRate * 100).toFixed(1)}%\`);
        }
      })
      .on('mouseout', () => tooltip.style('opacity', 0));

    // Chart 4: Calibration Gap
    const calibrationChart = createDualChart(
      margin.top + (chartHeight + spacing) * 3,
      yScaleCalibration,
      '#9b59b6',
      'calibration',
      'Calibration Gap',
      d => d.testCalibrationGap || 0,
      d => d.trainCalibrationGap || 0
    );

    calibrationChart.append('rect')
      .attr('width', width)
      .attr('height', chartHeight)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .on('mousemove', function(event) {
        const [mouseX] = d3.pointer(event);
        const sample = Math.round(xScale.invert(mouseX));
        const dataPoint = data.find(d => d.sample >= sample);
        if (dataPoint) {
          tooltip
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(\`<strong>Sample \${dataPoint.sample}</strong><br/>Test Cal.Gap: \${(dataPoint.testCalibrationGap || 0).toFixed(3)}<br/>Train Cal.Gap: \${(dataPoint.trainCalibrationGap || 0).toFixed(3)}\`);
        }
      })
      .on('mouseout', () => tooltip.style('opacity', 0));

    // X axis (only on bottom chart)
    calibrationChart.append('g')
      .attr('transform', \`translate(0,\${chartHeight})\`)
      .call(d3.axisBottom(xScale))
      .append('text')
      .attr('class', 'axis-label')
      .attr('x', width / 2)
      .attr('y', 40)
      .attr('fill', 'black')
      .text('Sample');
  </script>
</body>
</html>`;

  // Write HTML file
  const outputPath = join(runDir, "visualization.html");
  await writeFile(outputPath, html, "utf-8");

  console.log(`✓ Visualization generated: ${outputPath}`);
  console.log(`  Open in browser to view the chart`);
}

// CLI usage
if (import.meta.main) {
  const runDir = process.argv[2];
  if (!runDir) {
    console.error("Usage: bun src/v2/harness/visualize.ts <path-to-run-directory>");
    process.exit(1);
  }

  generateVisualization(runDir).catch((error) => {
    console.error("Error generating visualization:", error);
    process.exit(1);
  });
}

export { generateVisualization };
