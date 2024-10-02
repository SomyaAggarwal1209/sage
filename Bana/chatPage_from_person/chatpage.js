// ChatPage.js
import React, { useState } from 'react';
import Sidebar from './components/sideBar/side_bar';
import GroupChat from './components/groupChat/groupchat';
import IndividualChat from './components/individualChat/individual_chat';
import VideoCall from './components/videoCall/video_call';
import AudioCall from './components/audioCall/audio_call';
import './ChatPage.css'

const ChatPage = () => {
    const [activeTab, setActiveTab] = useState('individual');

    const renderActiveTab = () => {
        switch (activeTab) {
            case 'group':
                return <GroupChat />;
            case 'individual':
                return <IndividualChat />;
            case 'video':
                return <VideoCall />;
            case 'audio':
                return <AudioCall />;
            default:
                return <IndividualChat />;
        }
    };

    return (
        <div className="chat-page">
            <section className="section-chat-sidebar">
                <Sidebar setActiveTab={setActiveTab} />
            </section>
            <section className="section-chat-main-content">
                <div className="chat-content">
                    {renderActiveTab()}
                </div>
            </section>

        </div>
    );
};

export default ChatPage;
