import React, { useState } from "react";
import './VideoCall.css';
import { useNavigate } from "react-router-dom"; // Removed Routes and Route
import Room from "./room"; // Assuming you have this component

const VideoCall = () => {
    const [roomId, setRoomId] = useState("");
    const navigate = useNavigate();

    const handleRoomIdGenerate = () => {
        const randomId = Math.random().toString(36).substring(2, 9);
        const timestamp = Date.now().toString().slice(-4); // Corrected timestamp generation
        setRoomId(randomId + timestamp);
    };

    const handleOneAndOneCall = () => {
        if (!roomId) {
            alert("Please Generate Room Id First");
            return;
        }
        navigate(`/room/${roomId}?type=one-on-one`); // Added `/` to make the path absolute
    };

    const handleGroupCall = () => {
        if (!roomId) {
            alert("Please Generate Room Id First");
            return;
        }
        navigate(`/room/${roomId}?type=group-call`); // Added `/` to make the path absolute
    };

    return (
        <div className="video-call">
            <h1 className="homepage-title">Welcome to Video Calling App</h1>
            <p className="homepage-subtitle">
                Start a video call with a randomly generated Room ID
            </p>
            <div className="room-id-container">
                <input
                    type="text"
                    className="room-id-input"
                    placeholder="Generated Room ID"
                    value={roomId}
                    readOnly
                />
                <button className="generate-button" onClick={handleRoomIdGenerate}>
                    Generate
                </button>
            </div>
            <div className="call-buttons">
                <button
                    className="call-button"
                    onClick={handleOneAndOneCall}
                    disabled={!roomId}
                >
                    One-on-One Call
                </button>
                <button
                    className="call-button"
                    onClick={handleGroupCall}
                    disabled={!roomId}
                >
                    Group Call
                </button>
            </div>
        </div>
    );
};

export default VideoCall;
