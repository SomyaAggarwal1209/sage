// Sidebar.js
import React from 'react';
import './Sidebar.css';

const Sidebar = ({ setActiveGame }) => {
    return (
        <div className="sidebar">
            <h2>Study Games</h2>
            <ul>
                <li onClick={() => setActiveGame('memory')}>Memory Game</li>
                <li onClick={() => setActiveGame('quiz')}>Quiz Game</li>
                <li onClick={() => setActiveGame('typing')}>Typing Speed Game</li>
                <li onClick={() => setActiveGame('more')}>More Game</li>
            </ul>
        </div>
    );
};

export default Sidebar;
