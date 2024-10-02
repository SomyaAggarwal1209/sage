import React, { useState } from 'react';
import './a.css'; // Importing super cool CSS

const AssignmentGenerator = () => {
    const [subject, setSubject] = useState('');
    const [topic, setTopic] = useState('');
    const [questions, setQuestions] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubjectChange = (e) => {
        setSubject(e.target.value);
    };

    const handleTopicChange = (e) => {
        setTopic(e.target.value);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setQuestions([]);

        const input = {
            subject: subject,
            topic: topic,
        };

        try {
            const response = await fetch("https://assignments-0j5d.onrender.com/generate-assignment", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(input),
            });

            if (response.ok) {
                const data = await response.json();
                setQuestions(data.questions);
            } else {
                setError("Failed to fetch questions. Please try again.");
            }
        } catch (err) {
            setError("An error occurred. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="assignment-generator-container">
            <h1 className="title">Generate Assignment</h1>
            <form onSubmit={handleSubmit} className="assignment-form">
                <div className="dropdown-container">
                    <label htmlFor="subject">Select Subject:</label>
                    <select
                        id="subject"
                        value={subject}
                        onChange={handleSubjectChange}
                        className="dropdown"
                        required
                    >
                        <option value="">-- Select Subject --</option>
                        <option value="Physics">Physics</option>
                        <option value="Chemistry">Chemistry</option>
                        <option value="Mathematics">Mathematics</option>
                    </select>
                </div>

                <div className="dropdown-container">
                    <label htmlFor="topic">Select Topic:</label>
                    <select
                        id="topic"
                        value={topic}
                        onChange={handleTopicChange}
                        className="dropdown"
                        required
                    >
                        <option value="">-- Select Topic --</option>
                        <option value="Rotational Motion">Rotational Motion</option>
                        <option value="Thermodynamics">Thermodynamics</option>
                        <option value="Kinematics">Kinematics</option>
                    </select>
                </div>

                <button type="submit" className="generate-button">
                    {loading ? "Generating..." : "Generate Assignment"}
                </button>
            </form>

            {error && <p className="error-message">{error}</p>}

            {questions.length > 0 && (
                <div className="questions-container">
                    <h2>Assignment Questions</h2>
                    <ul className="questions-list">
                        {questions.map((question, index) => (
                            <li key={index} className="question-item">
                                {question}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default AssignmentGenerator;
