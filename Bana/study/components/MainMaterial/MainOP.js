import React, { useState } from 'react';
import AudioPage from './components/studyAudio/main_face';
import WrittenMaterial from './components/studyMaterial/studyMaterial';
import VideoPage from './components/studyVideo/video';
import './op.css'; // We'll define the CSS here

const StudyOp = () => {
    const [activeTab, setActiveTab] = useState('audio'); // Default tab is 'audio'

    const renderContent = () => {
        switch (activeTab) {
            case 'audio':
                return <AudioPage />;
            case 'written':
                return <WrittenMaterial />;
            case 'video':
                return <VideoPage />;
            default:
                return <AudioPage />;
        }
    };

    return (
        <div className="study-op-container">
            <div className="sub-navbar">
                <button
                    className={`sub-nav-button ${activeTab === 'audio' ? 'active' : ''}`}
                    onClick={() => setActiveTab('audio')}
                >
                    Study Audio
                </button>
                <button
                    className={`sub-nav-button ${activeTab === 'written' ? 'active' : ''}`}
                    onClick={() => setActiveTab('written')}
                >
                    Written Material
                </button>
                <button
                    className={`sub-nav-button ${activeTab === 'video' ? 'active' : ''}`}
                    onClick={() => setActiveTab('video')}
                >
                    Study Videos
                </button>
            </div>

            <div className="content-container">
                {renderContent()}
            </div>
        </div>
    );
};

export default StudyOp;
