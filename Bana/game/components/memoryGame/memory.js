// MemoryGame.js
import React, { useState } from 'react';
import './MemoryGame.css';

const MemoryGame = () => {
    const [cards, setCards] = useState([
        { id: 1, name: 'A', flipped: false },
        { id: 2, name: 'A', flipped: false },
        { id: 3, name: 'B', flipped: false },
        { id: 4, name: 'B', flipped: false },
        // Add more cards as needed
    ]);
    const [selectedCards, setSelectedCards] = useState([]);

    const flipCard = (index) => {
        const newCards = [...cards];
        newCards[index].flipped = !newCards[index].flipped;
        setCards(newCards);
    };

    return (
        <div className="memory-game">
            <h2>Memory Matching Game</h2>
            <div className="card-grid">
                {cards.map((card, index) => (
                    <div
                        key={index}
                        className={`card ${card.flipped ? 'flipped' : ''}`}
                        onClick={() => flipCard(index)}
                    >
                        {card.flipped ? card.name : '?'}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default MemoryGame;
