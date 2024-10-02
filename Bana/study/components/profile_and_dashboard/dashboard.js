import React from 'react';
import './ProfileDashboard.css';

const ProfileDashboard = () => {
    const assignments = [
        { subject: 'Physics', topic: 'Rotational Motion', dueDate: '2024-10-15' },
        { subject: 'Math', topic: 'Calculus', dueDate: '2024-10-20' },
        { subject: 'Chemistry', topic: 'Organic Chemistry', dueDate: '2024-10-25' },
    ];

    const testProgress = [
        { name: 'Mock Test 1', score: 85, date: '2024-09-30' },
        { name: 'Mock Test 2', score: 90, date: '2024-09-28' },
    ];

    const profilePicUrl = 'https://picsum.photos/seed/picsum/150'; // Replace with your profile picture URL

    return (
        <div className="profile-dashboard">
            <h2 className="dashboard-title">User Profile</h2>
            <div className="grid-container">
                <div className="profile-info">
                    <img src={profilePicUrl} alt="Profile" className="profile-pic" />
                    <h3 className="profile-heading">Profile Details</h3>
                    <div className="profile-card">
                        <p><strong>Name:</strong> John Doe</p>
                        <p><strong>Current Subjects:</strong> 5</p>
                        <p><strong>Study Time This Week:</strong> 10 hours</p>
                        <p><strong>Progress:</strong> 70%</p>
                    </div>
                </div>
                <div className="assignments-container">
                    <h3 className="assignments-heading">Assignments</h3>
                    <ul className="assignments-list">
                        {assignments.map((assignment, index) => (
                            <li key={index} className="assignment-item">
                                <p><strong>Subject:</strong> {assignment.subject}</p>
                                <p><strong>Topic:</strong> {assignment.topic}</p>
                                <p><strong>Due Date:</strong> {assignment.dueDate}</p>
                            </li>
                        ))}
                    </ul>
                </div>
                <div className="test-progress-container">
                    <h3 className="test-progress-heading">Test Progress</h3>
                    <ul className="test-progress-list">
                        {testProgress.map((test, index) => (
                            <li key={index} className="test-progress-item">
                                <p><strong>{test.name}</strong> - Score: {test.score} - Date: {test.date}</p>
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default ProfileDashboard;
