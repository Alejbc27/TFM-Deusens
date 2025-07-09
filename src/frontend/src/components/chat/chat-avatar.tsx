import { Bot, User } from 'lucide-react';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { cn } from '@/lib/utils';

interface ChatAvatarProps {
  sender: 'user' | 'agent';
}

export function ChatAvatar({ sender }: ChatAvatarProps) {
  const isAgent = sender === 'agent';
  return (
    <Avatar className={cn(
      'w-10 h-10 border-2 flex items-center justify-center',
      isAgent ? 'border-primary/50' : 'border-accent/50'
    )}>
      <AvatarFallback className="bg-transparent">
        {isAgent ? (
          <Bot className="w-6 h-6 text-primary" style={{ filter: 'drop-shadow(0 0 5px hsl(var(--primary)))' }} />
        ) : (
          <User className="w-6 h-6 text-accent" style={{ filter: 'drop-shadow(0 0 5px hsl(var(--accent)))' }} />
        )}
      </AvatarFallback>
    </Avatar>
  );
}
