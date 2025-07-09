'use server';

export async function getAgentResponse(prompt: string): Promise<string> {
  try {
    const response = await fetch('http://localhost:8081/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt }),
      signal: AbortSignal.timeout(10000), 
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
    return 'Error: Could not connect to the agent. Please ensure it is running on http://localhost:8081 and accessible.';
  }
}
