import { z } from "zod";

// Trie node schema
export const TrieNodeSchema = z.object({
  id: z.number().int().positive(),
  parent_id: z.number().int().positive().nullable(),
  char: z.string().length(1),
  pattern: z.string().nullable(),
  terminal: z.number().int().min(0).max(1).default(0),
  markov_id: z.number().int().positive().nullable(),
});

// Input schema for trie node insertion
export const TrieNodeInsertSchema = TrieNodeSchema.pick({
  parent_id: true,
  char: true,
  terminal: true,
});

// Update schema for terminal nodes
export const TrieNodeUpdateSchema = TrieNodeSchema.pick({
  id: true,
  pattern: true,
  markov_id: true,
}).extend({
  terminal: z.literal(1),
});

// Export types
export type TrieNode = z.infer<typeof TrieNodeSchema>;
export type TrieNodeInsert = z.infer<typeof TrieNodeInsertSchema>;
export type TrieNodeUpdate = z.infer<typeof TrieNodeUpdateSchema>;
