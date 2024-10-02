// Footer.js
import React from 'react';
import './Footer.css';

const Footer = () => {
    return (
        <footer className="footer">
            <div className="footer-content">
                <p className="footer-logo">Study Peace</p>
                <ul className="footer-links">
                    <li><a href="#privacy">Privacy Policy</a></li>
                    <li><a href="#terms">Terms of Service</a></li>
                    <li><a href="#support">Support</a></li>
                </ul>
            </div>
            <div className="footer-copyright">
                <p>&copy; 2024 Study Peace. All Rights Reserved.</p>
            </div>
        </footer>
    );
};

export default Footer;
