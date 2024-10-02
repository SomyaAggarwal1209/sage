// Courses.js
import React from 'react';
import './Courses.css';

const Courses = () => {
    const courses = [
        { name: 'Complete Python Course', link: '#' },
        { name: 'Data Science with R', link: '#' },
        // Add more courses
    ];

    return (
        <div className="courses">
            <h2>Available Courses</h2>
            <ul>
                {courses.map((course, index) => (
                    <li key={index}>
                        <a href={course.link}>{course.name}</a>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default Courses;
