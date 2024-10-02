import React from 'react';
import './navbar.css';

const Navbar = () => {
    return (
        <nav className="navbar">
            <div className="navbar-logo">Study Peace</div>
            <ul className="navbar-links">
                <li><a href="/">Home</a></li>
                <li><a href="/study-page">Study</a></li>
                <li><a href="/chat-page">Chat</a></li>
                <li><a href="/study-games-page">Games</a></li>
                <li><a href="/about">About & Contact</a></li>
            </ul>
        </nav>
    );
};

export default Navbar;
