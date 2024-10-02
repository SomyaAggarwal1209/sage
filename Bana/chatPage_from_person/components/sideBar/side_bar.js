// Sidebar.js
import React from 'react';
import './Sidebar.css';

const Sidebar = ({ setActiveTab }) => {
    return (
        <div className="sidebar">
            <h2>Options</h2>
            <ul>
                <li onClick={() => setActiveTab('individual')}>Individual Chat</li>
                <li onClick={() => setActiveTab('group')}>Group Chat</li>
                <li onClick={() => setActiveTab('video')}>Video Call</li>
                <li onClick={() => setActiveTab('audio')}>Queue Chat</li>
            </ul>
        </div>
    );
};

export default Sidebar;
