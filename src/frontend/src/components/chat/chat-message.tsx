import { cn } from '@/lib/utils';
import type { Message } from '@/types';
import { ChatAvatar } from './chat-avatar';

interface ChatMessageProps {
  message: Message;
  isLoading?: boolean;
}

export function ChatMessage({ message, isLoading = false }: ChatMessageProps) {
  const isAgent = message.sender === 'agent';
  return (
    <div
      className={cn(
        'flex items-start gap-3 animate-in fade-in zoom-in-95',
        isAgent ? 'justify-start' : 'justify-end'
      )}
    >
      {isAgent && <ChatAvatar sender="agent" />}
      <div
        className={cn(
          'relative max-w-md lg:max-w-xl px-4 py-3 rounded-2xl shadow-md',
          isAgent
            ? 'bg-secondary rounded-bl-none text-secondary-foreground'
            : 'bg-primary rounded-br-none text-primary-foreground',
          isLoading && 'bg-secondary'
        )}
      >
        {isLoading ? (
          <div className="flex items-center space-x-1 text-secondary-foreground">
            <span className="h-2 w-2 bg-current rounded-full animate-bounce [animation-delay:-0.3s]"></span>
            <span className="h-2 w-2 bg-current rounded-full animate-bounce [animation-delay:-0.15s]"></span>
            <span className="h-2 w-2 bg-current rounded-full animate-bounce"></span>
          </div>
        ) : (
          <p className="whitespace-pre-wrap">{message.content}</p>
        )}
      </div>
      {!isAgent && <ChatAvatar sender="user" />}
    </div>
  );
}
