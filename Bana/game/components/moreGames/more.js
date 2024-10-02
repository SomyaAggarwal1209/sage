import React from 'react';
import './more.css';

const BrainGames = () => {
    return (
        <div className="brain-games-container">
            <h2 className="brain-games-title">More Games</h2>
            <div className="iframe-container">
                <iframe
                    src="https://in.mathgames.com/"
                    title="Math Games"
                    className="mathgames-iframe"
                    frameBorder="0"
                    allowFullScreen
                ></iframe>
            </div>
        </div>
    );
};

export default BrainGames;
