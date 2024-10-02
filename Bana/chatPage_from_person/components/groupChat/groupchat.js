import React, { useState } from 'react';
import './GroupChat.css';

const GroupChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [selectedGroup, setSelectedGroup] = useState(null);

  // Sample list of groups
  const groups = [
    { id: 1, name: 'Project Team' },
    { id: 2, name: 'Friends' },
    { id: 3, name: 'Family' },
  ];

  // Handle group selection
  const handleSelectGroup = (group) => {
    setSelectedGroup(group);
    setMessages([]); // Clear messages when starting a new group chat
  };

  // Handle sending a message
  const handleSendMessage = () => {
    if (input.trim() !== '') {
      setMessages([...messages, { text: input, sender: 'user' }]);
      setInput('');

      // Simulate a group member's reply
      setTimeout(() => {
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: `Hi, this is a reply in ${selectedGroup.name} group.`, sender: 'member' },
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
      {/* Sidebar with group list */}
      <div className="group-list">
        <h3>Groups</h3>
        <ul>
          {groups.map((group) => (
            <li
              key={group.id}
              className={selectedGroup && selectedGroup.id === group.id ? 'active' : ''}
              onClick={() => handleSelectGroup(group)}
            >
              {group.name}
            </li>
          ))}
        </ul>
      </div>

      {/* Chat section */}
      <div className="chat-section">
        {selectedGroup ? (
          <>
            <h2>{selectedGroup.name} Group Chat</h2>
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
          <div className="no-group-selected">
            <h2>Select a group to start chatting</h2>
          </div>
        )}
      </div>
    </div>
  );
};

export default GroupChat;
