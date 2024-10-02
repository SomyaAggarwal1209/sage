import React, { useState } from 'react';
import './StudyMusic.css';

const StudyMusic = () => {
    const [currentMusic, setCurrentMusic] = useState(null);

    const musicTracks = [
        { title: 'Calm Focus', file: 'calm-focus.mp3', image: 'https://picsum.photos/seed/picsum/150' }, // Placeholder image URL
        { title: 'Deep Study', file: 'deep-study.mp3', image: 'https://picsum.photos/seed/picsum/150' }, // Placeholder image URL
        { title: 'Nature Sounds', file: 'nature-sounds.mp3', image: 'https://picsum.photos/seed/picsum/150' }, // Placeholder image URL
        // Add more music tracks with images
    ];

    const handlePlay = (file) => {
        const music = new Audio(file);
        if (currentMusic) currentMusic.pause(); // Pause the previous track
        music.play();
        setCurrentMusic(music);
    };

    return (
        <div className="study-music-container">
            <h2 className="study-music-title">Study Music</h2>
            <div className="study-music">
                {musicTracks.map((track, index) => (
                    <div className="music-card" key={index}>
                        <img src={track.image} alt={track.title} className="music-image" />
                        <div className="music-info">
                            <h3>{track.title}</h3>
                            <button className="play-button" onClick={() => handlePlay(track.file)}>Play</button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default StudyMusic;
