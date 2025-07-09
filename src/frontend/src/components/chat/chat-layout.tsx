import { ChatHeader } from './chat-header';
import { ChatMessages } from './chat-messages';
import { ChatInput } from './chat-input';
import type { Message } from '@/types';
import type { FormEvent } from 'react';

interface ChatLayoutProps {
  messages: Message[];
  input: string;
  setInput: (value: string) => void;
  handleSubmit: (e: FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
  aiTip: string | null;
  fetchAiTip: (content: string) => void;
}

export function ChatLayout({ messages, input, setInput, handleSubmit, isLoading, aiTip, fetchAiTip }: ChatLayoutProps) {
  return (
    <div className="relative z-10 flex flex-col w-full max-w-4xl h-[95dvh] max-h-[900px] border rounded-2xl shadow-2xl bg-card/50 backdrop-blur-md border-primary/20 shadow-primary/20">
      <ChatHeader />
      <ChatMessages messages={messages} isLoading={isLoading} />
      <ChatInput
        input={input}
        setInput={setInput}
        handleSubmit={handleSubmit}
        isLoading={isLoading}
        aiTip={aiTip}
        fetchAiTip={fetchAiTip}
      />
    </div>
  );
}
