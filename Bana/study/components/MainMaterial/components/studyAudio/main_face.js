import React, { useState } from 'react';
import StudyAudio from "./components/study_audio/audio";
import StudyMusic from "./components/study_music/music";
import './see.css';

const StudyMusicandAudio = () => {
    const [activeTab, setActiveTab] = useState('audio'); // Default tab is 'audio'

    const renderContent = () => {
        switch (activeTab) {
            case 'audio':
                return <StudyAudio />;
            case 'music':
                return <StudyMusic />;
            default:
                return <StudyAudio />;
        }
    };

    return (
        <div className="study-music-audio-container">
            <div className="sub-navbar">
                <button
                    className={`sub-nav-button ${activeTab === 'audio' ? 'active' : ''}`}
                    onClick={() => setActiveTab('audio')}
                >
                    Study Audio
                </button>
                <button
                    className={`sub-nav-button ${activeTab === 'music' ? 'active' : ''}`}
                    onClick={() => setActiveTab('music')}
                >
                    Study Music
                </button>
            </div>

            <div className="content-container">
                {renderContent()}
            </div>
        </div>
    );
}

export default StudyMusicandAudio;
