// TypingSpeedGame.js
import React, { useState, useEffect } from 'react';
import './TypingSpeedGame.css';

const sampleText = 'The quick brown fox jumps over the lazy dog.';

const TypingSpeedGame = () => {
    const [inputText, setInputText] = useState('');
    const [startTime, setStartTime] = useState(null);
    const [typingSpeed, setTypingSpeed] = useState(null);

    useEffect(() => {
        if (inputText.length === sampleText.length) {
            const endTime = new Date();
            const timeTaken = (endTime - startTime) / 1000; // Time in seconds
            const wordsPerMinute = (sampleText.split(' ').length / timeTaken) * 60;
            setTypingSpeed(wordsPerMinute);
        }
    }, [inputText, startTime]);

    const handleInputChange = (e) => {
        if (!startTime) setStartTime(new Date());
        setInputText(e.target.value);
    };

    return (
        <div className="typing-speed-game">
            <h2>Typing Speed Game</h2>
            <p>Type the following sentence as quickly as you can:</p>
            <p className="sample-text">{sampleText}</p>
            <textarea
                value={inputText}
                onChange={handleInputChange}
                rows="4"
                placeholder="Start typing..."
            />
            {typingSpeed && <h3>Your typing speed: {Math.round(typingSpeed)} WPM</h3>}
        </div>
    );
};

export default TypingSpeedGame;
