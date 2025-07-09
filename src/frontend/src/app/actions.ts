'use server';

import type { Message } from '@/types';
import { v4 as uuidv4 } from 'uuid';

export async function getAgentResponse(prompt: string, threadId: string): Promise<string> {
  try {
    const response = await fetch('http://agent-api:8081/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: prompt, thread_id: threadId }),
      signal: AbortSignal.timeout(30000), 
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Agent API error:', response.status, errorText);
      return `Error from agent: ${response.status} ${response.statusText}. The agent might be offline or encountering an issue.`;
    }
    
    const responseData = await response.text();
    try {
      const json = JSON.parse(responseData);
      return json.response || json.text || json.message || responseData;
    } catch (e) {
      return responseData; 
    }

  } catch (error: any) {
    console.error('Error calling agent API:', error);
    if (error.name === 'AbortError') {
        return 'Error: Connection to the agent timed out.';
    }
    return 'Error: Could not connect to the agent. Please ensure it is running on http://localhost:3100 and accessible.';
  }
}

export async function getChatHistory(threadId: string): Promise<Message[]> {
  try {
    const response = await fetch(`http://agent-api:8081/sessions/${threadId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(10000), 
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Agent API history error:', response.status, errorText);
      return [];
    }

    const historyResponse = await response.json();
    
    if (historyResponse && historyResponse.history) {
      // Map backend messages to frontend Message interface
      const transformedHistory: Message[] = historyResponse.history.map((msg: any) => {
        let sender: 'user' | 'agent';
        if (msg.type === 'human') {
          sender = 'user';
        } else if (msg.type === 'ai') {
          sender = 'agent';
        } else {
          sender = 'agent'; // Default or handle other types as needed
        }
        return {
          id: uuidv4(), // Generate a unique ID for each message
          content: msg.content || '', // Ensure content exists
          sender: sender,
        };
      });
      return transformedHistory;
    } else {
      return [];
    }
  } catch (error: any) {
    console.error('Error fetching chat history:', error);
    return [];
  }
}