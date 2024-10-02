import React, { useState } from 'react';
import './choose_bar.css';

const ChooseResource = () => {
    const [selectedResource, setSelectedResource] = useState('');

    const handleSelect = (resource) => {
        setSelectedResource(resource);
        alert(`You selected: ${resource}`);
    };

    return (
        <div className="choose-resource-container">
            <h1 className="resource-title">Choose Your Resource</h1>
            <div className="resource-options">
                <div
                    className={`resource-option ${selectedResource === 'book' ? 'selected' : ''}`}
                    onClick={() => handleSelect('book')}
                >
                    <h2>Book</h2>
                    <p>Explore our collection of insightful books.</p>
                </div>
                <div
                    className={`resource-option ${selectedResource === 'course' ? 'selected' : ''}`}
                    onClick={() => handleSelect('course')}
                >
                    <h2>Course</h2>
                    <p>Join our interactive courses for in-depth learning.</p>
                </div>
            </div>
        </div>
    );
};

export default ChooseResource;
