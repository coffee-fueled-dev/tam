/**
 * Generate self-contained HTML visualization for latent space evolution
 *
 * Usage:
 *   bun src/v2/harness/visualize-latent.ts <path-to-run-directory>
 */

import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

interface LatentCheckpoint {
  sample: number;
  ports: Array<{
    portIdx: number;
    embedding: number[];
    commitments: Array<{
      stateIdx: number;
      concentration: number;
      selected: boolean;
    }>;
  }>;
}

async function generateLatentVisualization(runDir: string) {
  // Read latent data
  const latentPath = join(runDir, "latent.jsonl");
  const latentText = await readFile(latentPath, "utf-8");
  const checkpoints: LatentCheckpoint[] = latentText
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));

  // Read metadata
  const metadataPath = join(runDir, "metadata.json");
  const metadata = JSON.parse(await readFile(metadataPath, "utf-8"));

  // Extract representative states from first checkpoint
  let representativeStates: number[] = [];
  if (checkpoints.length > 0 && checkpoints[0]!.ports.length > 0) {
    representativeStates = checkpoints[0]!.ports[0]!.commitments.map(c => c.stateIdx);
  }

  // Generate HTML with embedded data
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${metadata.experiment} - Latent Space</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 20px;
      background: #f5f5f5;
    }

    .container {
      max-width: 1400px;
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

    .controls {
      margin: 20px 0;
      padding: 15px;
      background: #f8f8f8;
      border-radius: 4px;
    }

    .chart-container {
      display: flex;
      gap: 30px;
      margin-top: 20px;
    }

    .chart {
      flex: 1;
    }

    .port-circle {
      stroke: #fff;
      stroke-width: 2;
    }

    .port-label {
      font-size: 10px;
      font-weight: bold;
      fill: #333;
      pointer-events: none;
    }

    .axis-label {
      font-size: 12px;
      font-weight: 600;
    }

    .heatmap-cell {
      stroke: #fff;
      stroke-width: 1;
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

    .slider {
      width: 100%;
      margin: 10px 0;
    }

    .checkpoint-info {
      font-weight: bold;
      color: #3498db;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>${metadata.experiment} - Latent Space</h1>

    <div class="metadata">
      <strong>Run ID:</strong> <code>${metadata.runId}</code><br>
      <strong>Timestamp:</strong> ${new Date(metadata.timestamp).toLocaleString()}<br>
      <strong>Checkpoints:</strong> ${checkpoints.length}
    </div>

    <div class="controls">
      <div>
        <label for="checkpoint-slider">Checkpoint: <span id="checkpoint-label" class="checkpoint-info"></span></label>
        <input type="range" id="checkpoint-slider" class="slider" min="0" max="${checkpoints.length - 1}" value="${checkpoints.length - 1}">
      </div>
    </div>

    <div class="chart-container">
      <div class="chart" id="embedding-chart"></div>
      <div class="chart" id="heatmap-chart"></div>
    </div>
    <div id="tooltip" class="tooltip"></div>
  </div>

  <script>
    const checkpoints = ${JSON.stringify(checkpoints, null, 2)};
    const representativeStates = ${JSON.stringify(representativeStates)};

    // PCA helper function with consistent orientation
    let lastPC1 = null;
    let lastPC2 = null;

    function pca2D(embeddings) {
      // Center the data
      const dim = embeddings[0].length;
      const means = [];
      for (let d = 0; d < dim; d++) {
        means.push(d3.mean(embeddings, emb => emb[d]));
      }

      const centered = embeddings.map(emb =>
        emb.map((val, d) => val - means[d])
      );

      // Compute covariance matrix
      const cov = [];
      for (let i = 0; i < dim; i++) {
        cov[i] = [];
        for (let j = 0; j < dim; j++) {
          let sum = 0;
          for (const c of centered) {
            sum += c[i] * c[j];
          }
          cov[i][j] = sum / centered.length;
        }
      }

      // Power iteration to find top 2 eigenvectors
      function powerIteration(matrix, numIters = 20) {
        let v = Array(dim).fill(0).map(() => Math.random());
        let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        v = v.map(x => x / norm);

        for (let iter = 0; iter < numIters; iter++) {
          const Av = matrix.map(row =>
            row.reduce((sum, val, i) => sum + val * v[i], 0)
          );
          norm = Math.sqrt(Av.reduce((s, x) => s + x * x, 0));
          v = Av.map(x => x / norm);
        }

        return v;
      }

      let pc1 = powerIteration(cov);

      // Deflate to find second component
      const cov2 = cov.map((row, i) =>
        row.map((val, j) => val - pc1[i] * pc1[j] * (cov.map((r, k) => r[i] * pc1[k]).reduce((a, b) => a + b, 0)))
      );
      let pc2 = powerIteration(cov2);

      // Fix sign ambiguity: ensure consistent orientation across checkpoints
      if (lastPC1 !== null) {
        // Dot product with previous PC1 - flip if negative (opposite direction)
        const dot1 = pc1.reduce((sum, val, i) => sum + val * lastPC1[i], 0);
        if (dot1 < 0) {
          pc1 = pc1.map(v => -v);
        }
      }

      if (lastPC2 !== null) {
        // Dot product with previous PC2 - flip if negative
        const dot2 = pc2.reduce((sum, val, i) => sum + val * lastPC2[i], 0);
        if (dot2 < 0) {
          pc2 = pc2.map(v => -v);
        }
      }

      // Store for next checkpoint
      lastPC1 = pc1;
      lastPC2 = pc2;

      // Project onto PC1 and PC2
      return centered.map(c => [
        c.reduce((sum, val, i) => sum + val * pc1[i], 0),
        c.reduce((sum, val, i) => sum + val * pc2[i], 0)
      ]);
    }

    // Slider control
    document.getElementById('checkpoint-slider').addEventListener('input', function(e) {
      visualize(parseInt(e.target.value));
    });

    // Initialize with last checkpoint
    visualize(checkpoints.length - 1);

    function visualize(checkpointIdx) {
      const checkpoint = checkpoints[checkpointIdx];
      document.getElementById('checkpoint-label').textContent =
        \`Sample \${checkpoint.sample} (\${checkpoint.ports.length} ports)\`;

      // Clear previous charts
      d3.select('#embedding-chart').html('');
      d3.select('#heatmap-chart').html('');

      visualizeEmbeddings(checkpoint);
      visualizeHeatmap(checkpoint);
    }

    function visualizeEmbeddings(checkpoint) {
      const margin = { top: 40, right: 20, bottom: 60, left: 60 };
      const width = 600 - margin.left - margin.right;
      const height = 500 - margin.top - margin.bottom;

      const svg = d3.select('#embedding-chart')
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', \`translate(\${margin.left},\${margin.top})\`);

      // Check dimensionality and apply PCA if needed
      const allEmbeddings = checkpoint.ports.map(p => p.embedding);
      const embeddingDim = allEmbeddings[0].length;
      let projectedEmbeddings;
      let usedPCA = false;

      if (embeddingDim > 2) {
        projectedEmbeddings = pca2D(allEmbeddings);
        usedPCA = true;
      } else {
        projectedEmbeddings = allEmbeddings;
      }

      // Title
      const titleText = usedPCA
        ? \`Port Embeddings (PCA: \${embeddingDim}D → 2D)\`
        : 'Port Embeddings with Cone Ranges (Polysemanticity)';
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', -15)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .text(titleText);

      // Find extent of projected embeddings
      const xExtent = d3.extent(projectedEmbeddings, d => d[0]);
      const yExtent = d3.extent(projectedEmbeddings, d => d[1]);

      const xScale = d3.scaleLinear()
        .domain([xExtent[0] - 0.5, xExtent[1] + 0.5])
        .range([0, width]);

      const yScale = d3.scaleLinear()
        .domain([yExtent[0] - 0.5, yExtent[1] + 0.5])
        .range([height, 0]);

      // Color scale for ports
      const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

      // Axes
      const xAxisLabel = usedPCA ? 'PC1' : 'Dimension 1 (e.g., position)';
      const yAxisLabel = usedPCA ? 'PC2' : 'Dimension 2 (e.g., velocity)';

      svg.append('g')
        .attr('transform', \`translate(0,\${height})\`)
        .call(d3.axisBottom(xScale))
        .append('text')
        .attr('class', 'axis-label')
        .attr('x', width / 2)
        .attr('y', 40)
        .attr('fill', 'black')
        .text(xAxisLabel);

      svg.append('g')
        .call(d3.axisLeft(yScale))
        .append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -45)
        .attr('fill', 'black')
        .attr('text-anchor', 'middle')
        .text(yAxisLabel);

      const tooltip = d3.select('#tooltip');

      // Draw ports with anisotropic cones (ellipses)
      checkpoint.ports.forEach((port, idx) => {
        const projEmb = projectedEmbeddings[idx];

        // Compute average cone radius across states
        const avgConeRadius = [0, 0];
        for (const commit of port.commitments) {
          avgConeRadius[0] += commit.coneRadius[0] || 0;
          avgConeRadius[1] += commit.coneRadius[1] || 0;
        }
        avgConeRadius[0] /= port.commitments.length;
        avgConeRadius[1] /= port.commitments.length;

        // Compute min/max cone radii for visualization range
        const minRadius = [
          d3.min(port.commitments, c => c.coneRadius[0] || 0.01),
          d3.min(port.commitments, c => c.coneRadius[1] || 0.01)
        ];
        const maxRadius = [
          d3.max(port.commitments, c => c.coneRadius[0] || 0.01),
          d3.max(port.commitments, c => c.coneRadius[1] || 0.01)
        ];

        // Scale factor for visualization (map cone radius to pixels)
        const radiusScale = 50;

        // Draw max cone (widest - faint ellipse)
        svg.append('ellipse')
          .attr('class', 'port-circle')
          .attr('cx', xScale(projEmb[0]))
          .attr('cy', yScale(projEmb[1]))
          .attr('rx', Math.min(maxRadius[0] * radiusScale, 60))
          .attr('ry', Math.min(maxRadius[1] * radiusScale, 60))
          .attr('fill', colorScale(port.portIdx))
          .attr('opacity', 0.1)
          .attr('stroke', 'none');

        // Draw min cone (narrowest - bold ellipse)
        svg.append('ellipse')
          .attr('class', 'port-circle')
          .attr('cx', xScale(projEmb[0]))
          .attr('cy', yScale(projEmb[1]))
          .attr('rx', Math.min(minRadius[0] * radiusScale, 60))
          .attr('ry', Math.min(minRadius[1] * radiusScale, 60))
          .attr('fill', colorScale(port.portIdx))
          .attr('opacity', 0.6)
          .on('mouseover', function(event) {
            const origEmbStr = port.embedding.map(v => v.toFixed(2)).join(', ');
            const projEmbStr = usedPCA
              ? \`<br/>Projected: [\${projEmb.map(v => v.toFixed(2)).join(', ')}]\`
              : '';
            tooltip
              .style('opacity', 1)
              .style('left', (event.pageX + 10) + 'px')
              .style('top', (event.pageY - 10) + 'px')
              .html(\`
                <strong>Port \${port.portIdx}</strong><br/>
                Embedding: [\${origEmbStr}]\${projEmbStr}<br/>
                Min cone radius: [\${minRadius.map(r => r.toFixed(3)).join(', ')}]<br/>
                Max cone radius: [\${maxRadius.map(r => r.toFixed(3)).join(', ')}]<br/>
                Avg cone radius: [\${avgConeRadius.map(r => r.toFixed(3)).join(', ')}]<br/>
                <em>Ellipse shows anisotropy</em>
              \`);
          })
          .on('mouseout', () => tooltip.style('opacity', 0));

        svg.append('text')
          .attr('class', 'port-label')
          .attr('x', xScale(projEmb[0]))
          .attr('y', yScale(projEmb[1]))
          .attr('dy', 4)
          .attr('text-anchor', 'middle')
          .text(port.portIdx);
      });
    }

    function visualizeHeatmap(checkpoint) {
      const margin = { top: 40, right: 20, bottom: 60, left: 60 };
      const width = 600 - margin.left - margin.right;
      const height = 500 - margin.top - margin.bottom;

      const svg = d3.select('#heatmap-chart')
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', \`translate(\${margin.left},\${margin.top})\`);

      // Title
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', -15)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .text('Port Commitment Concentrations at Test States');

      const numPorts = checkpoint.ports.length;
      const numStates = representativeStates.length;

      const cellWidth = width / numStates;
      const cellHeight = height / numPorts;

      // Extract concentrations for color scale
      const allConcentrations = checkpoint.ports.flatMap(p => p.commitments.map(c => c.concentration));
      const colorScale = d3.scaleSequential(d3.interpolateYlGn)
        .domain([0, d3.max(allConcentrations)]);

      const tooltip = d3.select('#tooltip');

      // Draw heatmap
      checkpoint.ports.forEach((port, portIdx) => {
        port.commitments.forEach(commit => {
          svg.append('rect')
            .attr('class', 'heatmap-cell')
            .attr('x', commit.stateIdx * cellWidth)
            .attr('y', portIdx * cellHeight)
            .attr('width', cellWidth)
            .attr('height', cellHeight)
            .attr('fill', colorScale(commit.concentration))
            .attr('stroke-width', commit.selected ? 3 : 1)
            .attr('stroke', commit.selected ? '#000' : '#fff')
            .on('mouseover', function(event) {
              tooltip
                .style('opacity', 1)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px')
                .html(\`
                  <strong>Port \${port.portIdx} @ State \${commit.stateIdx}</strong><br/>
                  Concentration: \${commit.concentration.toFixed(3)}<br/>
                  Cone: \${(1/commit.concentration).toFixed(3)} (radius)<br/>
                  Selected: \${commit.selected ? 'Yes' : 'No'}
                \`);
            })
            .on('mouseout', () => tooltip.style('opacity', 0));
        });
      });

      // X axis (states)
      svg.append('g')
        .attr('transform', \`translate(0,\${height})\`)
        .call(d3.axisBottom(d3.scaleLinear().domain([0, numStates]).range([0, width])).ticks(numStates))
        .append('text')
        .attr('class', 'axis-label')
        .attr('x', width / 2)
        .attr('y', 40)
        .attr('fill', 'black')
        .text('Test State Index');

      // Y axis (ports)
      svg.append('g')
        .call(d3.axisLeft(d3.scaleLinear().domain([0, numPorts]).range([0, height])).ticks(numPorts))
        .append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -45)
        .attr('fill', 'black')
        .attr('text-anchor', 'middle')
        .text('Port Index');

      // Color legend
      const legendWidth = 200;
      const legendHeight = 20;
      const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', \`translate(\${width - legendWidth}, \${height + 35})\`);

      const legendScale = d3.scaleLinear()
        .domain([0, d3.max(allConcentrations)])
        .range([0, legendWidth]);

      const legendAxis = d3.axisBottom(legendScale).ticks(5);

      // Gradient
      const defs = svg.append('defs');
      const gradient = defs.append('linearGradient')
        .attr('id', 'legend-gradient');

      gradient.selectAll('stop')
        .data(d3.range(0, 1.1, 0.1))
        .enter().append('stop')
        .attr('offset', d => \`\${d * 100}%\`)
        .attr('stop-color', d => colorScale(d * d3.max(allConcentrations)));

      legend.append('rect')
        .attr('width', legendWidth)
        .attr('height', legendHeight)
        .style('fill', 'url(#legend-gradient)');

      legend.append('g')
        .attr('transform', \`translate(0,\${legendHeight})\`)
        .call(legendAxis);

      legend.append('text')
        .attr('x', legendWidth / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('Concentration (higher = narrower cone)');
    }
  </script>
</body>
</html>`;

  // Write HTML file
  const outputPath = join(runDir, "latent-visualization.html");
  await writeFile(outputPath, html, "utf-8");

  console.log(`✓ Latent visualization generated: ${outputPath}`);
}

// CLI usage
if (import.meta.main) {
  const runDir = process.argv[2];
  if (!runDir) {
    console.error("Usage: bun src/v2/harness/visualize-latent.ts <path-to-run-directory>");
    process.exit(1);
  }

  generateLatentVisualization(runDir).catch((error) => {
    console.error("Error generating latent visualization:", error);
    process.exit(1);
  });
}

export { generateLatentVisualization };
