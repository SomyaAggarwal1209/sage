import React, { useState } from 'react';
import './IndividualChat.css';

const IndividualChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [selectedContact, setSelectedContact] = useState(null);

  // List of contacts
  const contacts = [
    { id: 1, name: 'Alice' },
    { id: 2, name: 'Bob' },
    { id: 3, name: 'Charlie' },
  ];

  // Handle contact selection
  const handleSelectContact = (contact) => {
    setSelectedContact(contact);
    setMessages([]); // Clear messages when starting a new chat
  };

  // Function to handle sending a message
  const handleSendMessage = () => {
    if (input.trim() !== '') {
      setMessages([...messages, { text: input, sender: 'user' }]);
      setInput('');

      // Simulate a reply from the contact
      setTimeout(() => {
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: `Hi, this is ${selectedContact.name}'s reply.`, sender: 'friend' },
        ]);
      }, 1000);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="chat-container">
      {/* Sidebar with contact list */}
      <div className="contact-list">
        <h3>Contacts</h3>
        <ul>
          {contacts.map((contact) => (
            <li
              key={contact.id}
              className={selectedContact && selectedContact.id === contact.id ? 'active' : ''}
              onClick={() => handleSelectContact(contact)}
            >
              {contact.name}
            </li>
          ))}
        </ul>
      </div>

      {/* Chat section */}
      <div className="chat-section">
        {selectedContact ? (
          <>
            <h2>Chat with {selectedContact.name}</h2>
            <div className="messages">
              {messages.map((msg, index) => (
                <div key={index} className={`message ${msg.sender}`}>
                  {msg.text}
                </div>
              ))}
            </div>
            <div className="input-container">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type a message"
              />
              <button onClick={handleSendMessage}>Send</button>
            </div>
          </>
        ) : (
          <div className="no-contact-selected">
            <h2>Select a contact to start chatting</h2>
          </div>
        )}
      </div>
    </div>
  );
};

export default IndividualChat;
