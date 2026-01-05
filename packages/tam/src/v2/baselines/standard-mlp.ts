/**
 * Standard MLP Baseline
 *
 * Neural network that predicts mean only (no uncertainty).
 * Used as baseline to compare against TAM's dual-head approach.
 *
 * Architecture: Same hidden layers as CausalNet, but single output head.
 * Training: Standard MSE loss, no binding weighting.
 */

import * as tf from "@tensorflow/tfjs";
import type { Vec } from "../types";
import type Queue from "queue";

export interface StandardMLPConfig {
  hiddenSizes?: number[];
  learningRate?: number;
  batchSize?: number;
}

interface TrainingExample {
  stateEmb: Vec;
  portEmb: Vec;
  trajectory: Vec;
}

export class StandardMLP {
  private model: tf.LayersModel | null = null;
  private trainingBuffer: TrainingExample[] = [];
  private config: Required<StandardMLPConfig>;
  private queue: Queue;

  constructor(queue: Queue, config?: StandardMLPConfig) {
    this.queue = queue;
    this.config = {
      hiddenSizes: config?.hiddenSizes ?? [32, 32],
      learningRate: config?.learningRate ?? 0.001,
      batchSize: config?.batchSize ?? 10,
    };
  }

  /**
   * Predict trajectory (mean only, no uncertainty).
   */
  predict(stateEmb: Vec, portEmb: Vec): Vec {
    if (!this.model) {
      // No model yet - return zero prediction
      return Array(stateEmb.length).fill(0);
    }

    const input = [...stateEmb, ...portEmb];
    const inputTensor = tf.tensor2d([input]);
    const output = this.model.predict(inputTensor) as tf.Tensor;

    const result = output.arraySync() as number[][];
    inputTensor.dispose();
    output.dispose();

    return result[0]!;
  }

  /**
   * Queue a training example.
   */
  async observe(stateEmb: Vec, portEmb: Vec, trajectory: Vec): Promise<void> {
    this.trainingBuffer.push({
      stateEmb: [...stateEmb],
      portEmb: [...portEmb],
      trajectory: [...trajectory],
    });

    if (this.trainingBuffer.length >= this.config.batchSize) {
      await this.train();
    }
  }

  /**
   * Train on buffered examples with standard MSE loss.
   */
  async train(): Promise<void> {
    if (this.trainingBuffer.length === 0) return;

    const stateDim = this.trainingBuffer[0]!.stateEmb.length;
    const portDim = this.trainingBuffer[0]!.portEmb.length;
    const trajectoryDim = this.trainingBuffer[0]!.trajectory.length;

    // Build model if needed
    if (!this.model) {
      this.model = this.buildModel(stateDim + portDim, trajectoryDim);
    }

    // Prepare training data
    const inputs: number[][] = [];
    const targets: number[][] = [];

    for (const example of this.trainingBuffer) {
      inputs.push([...example.stateEmb, ...example.portEmb]);
      targets.push(example.trajectory);
    }

    const inputTensor = tf.tensor2d(inputs);
    const targetTensor = tf.tensor2d(targets);

    // Enqueue training to prevent concurrent operations
    await new Promise<void>((resolve, reject) => {
      this.queue.push(async () => {
        try {
          await this.model!.fit(inputTensor, targetTensor, {
            epochs: 1,
            verbose: 0,
          });

          inputTensor.dispose();
          targetTensor.dispose();

          resolve();
        } catch (error) {
          reject(error);
        }
      });
    });

    // Clear buffer
    this.trainingBuffer = [];
  }

  /**
   * Force flush remaining buffered data.
   */
  async flush(): Promise<void> {
    if (this.trainingBuffer.length > 0) {
      await this.train();
    }
  }

  /**
   * Build standard MLP (single output head for mean).
   */
  private buildModel(inputDim: number, outputDim: number): tf.LayersModel {
    const inputs = tf.input({ shape: [inputDim] });
    let x = inputs;

    // Hidden layers
    for (const size of this.config.hiddenSizes) {
      x = tf.layers.dense({ units: size, activation: "relu" }).apply(x) as tf.SymbolicTensor;
    }

    // Single output head (mean only)
    const output = tf.layers.dense({
      units: outputDim,
      activation: "linear",
      name: "mean",
    }).apply(x) as tf.SymbolicTensor;

    const model = tf.model({ inputs, outputs: output });

    // Standard MSE loss
    model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: "meanSquaredError",
    });

    return model;
  }

  /**
   * Clean up resources.
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.trainingBuffer = [];
  }
}
