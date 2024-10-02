import React, { useState } from 'react';
import './StudyAudio.css';

const StudyAudio = () => {
    const [currentAudio, setCurrentAudio] = useState(null);

    const audios = [
        { title: 'Lecture 1: Introduction to Chemistry', file: 'lecture1.mp3', image: 'https://picsum.photos/seed/picsum/150' }, // Placeholder image URL
        { title: 'Lecture 2: Physics Fundamentals', file: 'lecture2.mp3', image: 'https://picsum.photos/seed/picsum/150' }, // Placeholder image URL
        // Add more audio files with images
    ];

    const handlePlay = (file) => {
        const audio = new Audio(file);
        if (currentAudio) currentAudio.pause(); // Pause the previous audio
        audio.play();
        setCurrentAudio(audio);
    };

    return (
        <div className="study-audio-container">
            <h2 className="study-audio-title">Study Audio</h2>
            <div className="study-audio">
                {audios.map((audio, index) => (
                    <div className="audio-card" key={index}>
                        <img src={audio.image} alt={audio.title} className="audio-image" />
                        <div className="audio-info">
                            <h3>{audio.title}</h3>
                            <button className="play-button" onClick={() => handlePlay(audio.file)}>Play</button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default StudyAudio;
