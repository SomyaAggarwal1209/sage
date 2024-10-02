// AudioCall.js
import React from 'react';
import './AudioCall.css';

const AudioCall = () => {
    return (
        <div className="audio-call">
            <h2>Audio Call</h2>
            <div className="audio-screen">
                <p>Audio will be streamed here...</p>
            </div>
            <button className="end-call-button">End Call</button>
        </div>
    );
};

export default AudioCall;
