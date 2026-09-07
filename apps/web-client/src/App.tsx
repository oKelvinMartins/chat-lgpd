import { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Search, Bell, Zap, BookOpen, Shield, Send, MessageCircle } from 'lucide-react';
import './App.css';

interface Message {
  id: string;
  sender: 'user' | 'ai';
  text: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId] = useState(uuidv4());
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const isChatStarted = messages.length > 0;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = { id: uuidv4(), sender: 'user', text: inputValue };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    try {
      // Backend FastAPI local
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          user_id: 'guest_user',
          message: userMessage.text
        })
      });

      const data = await response.json();
      
      const aiMessage: Message = { id: uuidv4(), sender: 'ai', text: data.answer };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error("Erro ao chamar API", error);
      const errorMessage: Message = { id: uuidv4(), sender: 'ai', text: "Erro ao conectar com o servidor. O FastAPI está rodando?" };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Helper simples para quebras de linha
  const formatText = (text: string) => {
    return text.split('\n').map((line, i) => (
      <span key={i}>
        {line}
        <br />
      </span>
    ));
  };

  return (
    <>
      <div className="mesh-bg"></div>
      
      <div className="app-container">
        {/* Top Navigation */}
        <nav className="top-nav">
          <div className="logo-container">
            <div className="logo-icon">
              <Shield size={18} />
            </div>
            <span>LGPD AI</span>
          </div>
          
          <div className="nav-links">
            <span>Base Legal</span>
            <span>Casos de Uso</span>
            <span>Histórico</span>
          </div>
          
          <div className="nav-actions">
            <div className="search-pill">
              <Search size={16} />
              <span>Pesquisar artigos...</span>
            </div>
            <Bell size={20} color="var(--text-secondary)" style={{cursor: 'pointer'}} />
            <div className="avatar">
              <User size={18} />
            </div>
          </div>
        </nav>

        {/* Dynamic Main Area */}
        {!isChatStarted ? (
          <div className="center-stage">
            <div className="center-stage-logo">
              <Shield size={24} color="white" style={{marginTop: '12px'}} />
            </div>
            <h1>Olá, eu sou o LGPD AI</h1>
            <p>Seu oráculo pessoal sobre a Lei Geral de Proteção de Dados.</p>
          </div>
        ) : (
          <div className="chat-scroll-area">
            {messages.map(msg => (
              <div key={msg.id} className={`message-row ${msg.sender}`}>
                <div className={`bubble ${msg.sender}`}>
                  {formatText(msg.text)}
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="message-row ai">
                <div className="loading-indicator">
                  <div className="logo-icon" style={{width: 24, height: 24}}>
                    <MessageCircle size={14} />
                  </div>
                  Pesquisando na legislação e verificando fatos...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Input Area */}
        <div className={`input-wrapper ${!isChatStarted ? 'centered' : ''}`}>
          <div className="input-box">
            <textarea 
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Pergunte sobre direitos, multas, consentimento..."
              rows={1}
            />
            <button className="send-btn" onClick={handleSend} disabled={!inputValue.trim() || isTyping}>
              <Send size={18} />
            </button>
          </div>
          
          {!isChatStarted && (
            <div className="action-chips">
              <div className="chip" onClick={() => setInputValue("Quais são os meus direitos como titular?")}>
                <Zap size={14} /> Direitos
              </div>
              <div className="chip" onClick={() => setInputValue("Como funciona a multa da LGPD?")}>
                <BookOpen size={14} /> Multas
              </div>
              <div className="chip" onClick={() => setInputValue("Resuma as bases legais do artigo 7.")}>
                <Shield size={14} /> Bases Legais
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// User placeholder component since it's not exported by default in some lucide versions
function User(props: any) {
  return <svg xmlns="http://www.w3.org/2000/svg" width={props.size} height={props.size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
}

export default App;
