import { z } from "zod";

// Markov node schema
export const GraphNodeSchema = z.object({
  id: z.number().int().positive(),
  pattern: z.string(),
  confidence: z.number().default(0),
});

// Markov edge schema
export const GraphEdgeSchema = z.object({
  from_id: z.number().int().positive(),
  to_id: z.number().int().positive(),
  weight: z.number().positive().default(1),
});

// Input schema for node insertion
export const GraphNodeInsertSchema = GraphNodeSchema.pick({ pattern: true });

// Input schema for edge insertion
export const GraphEdgeInsertSchema = GraphEdgeSchema.pick({
  from_id: true,
  to_id: true,
}).extend({
  weight: z.number().positive().optional(),
});

// Export types
export type GraphNode = z.infer<typeof GraphNodeSchema>;
export type GraphEdge = z.infer<typeof GraphEdgeSchema>;
export type GraphNodeInsert = z.infer<typeof GraphNodeInsertSchema>;
export type GraphEdgeInsert = z.infer<typeof GraphEdgeInsertSchema>;
