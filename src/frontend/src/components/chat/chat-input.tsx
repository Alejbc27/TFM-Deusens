'use client';

import { useEffect, useRef } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Send, Lightbulb } from 'lucide-react';
import { useDebounce } from '@/hooks/use-debounce';
import type { FormEvent } from 'react';

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  handleSubmit: (e: FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
  aiTip: string | null;
  fetchAiTip: (content: string) => void;
}

export function ChatInput({ input, setInput, handleSubmit, isLoading, aiTip, fetchAiTip }: ChatInputProps) {
  const debouncedInput = useDebounce(input, 500);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchAiTip(debouncedInput);
  }, [debouncedInput, fetchAiTip]);
  
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <div className="p-4 border-t shrink-0 border-primary/20 bg-card/80 rounded-b-2xl">
      <div className="h-10 mb-2">
        {aiTip && !isLoading && (
            <div
              className="p-2 text-sm rounded-md bg-secondary/50 border border-secondary text-secondary-foreground flex items-center gap-2 animate-in fade-in slide-in-from-bottom-2"
            >
              <Lightbulb className="w-4 h-4 text-accent flex-shrink-0" style={{ filter: 'drop-shadow(0 0 3px hsl(var(--accent)))' }} />
              <span>{aiTip}</span>
            </div>
          )}
      </div>

      <form onSubmit={handleSubmit} className="flex items-center gap-3">
        <Input
          ref={inputRef}
          name="message"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter the void..."
          disabled={isLoading}
          className="flex-1 bg-background/50 border-2 border-primary/30 focus:border-primary focus:ring-2 focus:ring-primary/50 transition-all duration-300 h-12 text-base"
          autoComplete="off"
        />
        <Button
          type="submit"
          size="icon"
          disabled={isLoading || !input.trim()}
          className="w-12 h-12 rounded-full bg-primary hover:bg-primary/90 text-primary-foreground transition-all duration-300 transform hover:scale-110 disabled:scale-100 disabled:bg-primary/50"
          aria-label="Send message"
        >
          <Send className="w-6 h-6" />
        </Button>
      </form>
    </div>
  );
}
