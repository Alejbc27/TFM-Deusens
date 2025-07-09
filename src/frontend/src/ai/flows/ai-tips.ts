'use server';

/**
 * @fileOverview An AI-powered tips agent that provides guidance based on message content.
 *
 * - getAiTips - A function that generates AI tips based on the message content.
 * - AiTipsInput - The input type for the getAiTips function.
 * - AiTipsOutput - The return type for the getAiTips function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const AiTipsInputSchema = z.object({
  messageContent: z.string().describe('The content of the message to analyze.'),
});
export type AiTipsInput = z.infer<typeof AiTipsInputSchema>;

const AiTipsOutputSchema = z.object({
  tip: z.string().describe('An AI-powered tip based on the message content.'),
});
export type AiTipsOutput = z.infer<typeof AiTipsOutputSchema>;

export async function getAiTips(input: AiTipsInput): Promise<AiTipsOutput> {
  return aiTipsFlow(input);
}

const aiTipsPrompt = ai.definePrompt({
  name: 'aiTipsPrompt',
  input: {schema: AiTipsInputSchema},
  output: {schema: AiTipsOutputSchema},
  prompt: `You are an AI assistant designed to provide helpful tips based on the content of a user's message.

  Message Content: {{{messageContent}}}

  Based on the message content, provide a single, concise tip to help the user improve their interaction or communication. The tip should be actionable and relevant to the message's context.`,
});

const aiTipsFlow = ai.defineFlow(
  {
    name: 'aiTipsFlow',
    inputSchema: AiTipsInputSchema,
    outputSchema: AiTipsOutputSchema,
  },
  async input => {
    const {output} = await aiTipsPrompt(input);
    return output!;
  }
);
