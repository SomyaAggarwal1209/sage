// StudyGamesPage.js
import React, { useState } from 'react';
import Sidebar from './components/sideBar/side_bar';

import MemoryGame from './components/memoryGame/memory';
import QuizGame from './components/QuizGame/quiz_game';
import MoreGame from './components/moreGames/more'
import TypingSpeedGame from './components/typingGame/typing_game';
import './StudyGamesPage.css';


const StudyGamesPage = () => {
    const [activeGame, setActiveGame] = useState('memory');

    const renderActiveGame = () => {
        switch (activeGame) {
            case 'memory':
                return <MemoryGame />;
            case 'quiz':
                return <QuizGame />;
            case 'typing':
                return <TypingSpeedGame />;
            case 'more':
                return <MoreGame />;
            default:
                return <MemoryGame />;
        }
    };

    return (
        <div className="study-games-page">
            <section className="section-study-games-sidebar">
                <Sidebar setActiveGame={setActiveGame} />
            </section>
            <section className="section-study-games-game-content">
                <div className="game-content">
                    {renderActiveGame()}
                </div>
            </section>

        </div>
    );
};

export default StudyGamesPage;
