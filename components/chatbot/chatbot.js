import React, { useState } from 'react';
import './chatbot.css';

const Chatbot = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');

    const toggleChatbot = () => {
        setIsOpen(!isOpen);
    };

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (input.trim()) {
            // Add user's message to chat
            const newMessages = [...messages, { text: input, sender: 'user' }];
            setMessages(newMessages);

            // Prepare the query payload
            const queryPayload = {
                query: input,
            };

            try {
                // Send the message as a POST request
                const response = await fetch('https://project-7quz.onrender.com/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(queryPayload),
                });

                const data = await response.json();

                // Add the response message to chat
                setMessages([...newMessages, { text: data.response, sender: 'bot' }]);

            } catch (error) {
                console.error('Error sending message:', error);
                setMessages([...newMessages, { text: 'Error: Failed to get a response from the server.', sender: 'bot' }]);
            }

            setInput(''); // Clear input
        }
    };

    return (
        <div className={`chatbot-container ${isOpen ? 'open' : ''}`}>
            <div className="chatbot-header" onClick={toggleChatbot}>
                <p>Study Peace Chat</p>
            </div>
            {isOpen && (
                <div className="chatbot-body">
                    <div className="chatbot-messages">
                        {messages.map((msg, index) => (
                            <div key={index} className={`chatbot-message ${msg.sender}`}>
                                <p>{msg.text}</p>
                            </div>
                        ))}
                    </div>
                    <form className="chatbot-input-area" onSubmit={handleSendMessage}>
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Type a message..."
                        />
                        <button type="submit">Send</button>
                    </form>
                </div>
            )}
        </div>
    );
};

export default Chatbot;
