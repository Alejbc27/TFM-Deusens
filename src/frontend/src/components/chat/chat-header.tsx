import { Bot } from 'lucide-react';

export function ChatHeader() {
  return (
    <header className="flex items-center p-4 border-b shrink-0 border-primary/20">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 flex items-center justify-center">
            <Bot className="w-8 h-8 text-primary" style={{ filter: 'drop-shadow(0 0 5px hsl(var(--primary)))' }} />
        </div>
        <h1 className="text-2xl font-headline font-bold text-transparent bg-clip-text bg-gradient-to-r from-primary via-purple-300 to-white neon-text">
          NeonNexus Chat
        </h1>
      </div>
    </header>
  );
}
