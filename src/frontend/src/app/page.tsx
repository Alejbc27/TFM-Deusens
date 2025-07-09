'use client';

import { useState, useEffect } from 'react';
import type { FormEvent } from 'react';
import { ChatLayout } from '@/components/chat/chat-layout';
import { getAgentResponse, getChatHistory } from '@/app/actions';
import type { Message } from '@/types';
import { v4 as uuidv4 } from 'uuid';

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'agent-initial',
      sender: 'agent',
      content: 'Welcome to NeonNexus Chat! How can I assist you today in this digital realm?',
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [aiTip, setAiTip] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);

  useEffect(() => {
    let currentThreadId = localStorage.getItem('chatThreadId');
    if (!currentThreadId) {
      currentThreadId = uuidv4();
      localStorage.setItem('chatThreadId', currentThreadId);
    }
    setThreadId(currentThreadId);

    const fetchHistory = async () => {
      if (currentThreadId) {
        const history = await getChatHistory(currentThreadId);
        if (history.length > 0) {
          setMessages(history);
        }
      }
    };

    fetchHistory();
  }, []);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !threadId) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: input,
      sender: 'user',
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsLoading(true);
    setAiTip(null);

    const agentResponseContent = await getAgentResponse(currentInput, threadId);

    const agentMessage: Message = {
      id: `agent-${Date.now()}`,
      content: agentResponseContent,
      sender: 'agent',
    };

    setMessages((prev) => [...prev, agentMessage]);
    setIsLoading(false);
  };

  return (
    <main className="relative flex h-screen overflow-hidden">
      <div className="absolute inset-0 z-0 opacity-30">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_#4a0e70_0,_transparent_60%)]"></div>
        <div className="absolute top-0 left-0 w-1/2 h-1/2 bg-[radial-gradient(circle_at_top_left,_#be29ec_0,_transparent_50%)] animate-pulse-slow"></div>
        <div className="absolute bottom-0 right-0 w-1/2 h-1/2 bg-[radial-gradient(circle_at_bottom_right,_#ffd700_0,_transparent_50%)] animate-pulse-slow animation-delay-3000"></div>
      </div>
      <ChatLayout
        messages={messages}
        input={input}
        setInput={setInput}
        handleSubmit={handleSubmit}
        isLoading={isLoading}
        aiTip={aiTip}
      />
    </main>
  );
}
