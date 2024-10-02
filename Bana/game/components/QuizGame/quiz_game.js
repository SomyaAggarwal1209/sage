// QuizGame.js
import React, { useState } from 'react';
import './QuizGame.css';

const QuizGame = () => {
    const [questions] = useState([
        { question: 'What is the capital of France?', options: ['Paris', 'London', 'Berlin', 'Rome'], answer: 'Paris' },
        { question: 'What is 2 + 2?', options: ['3', '4', '5', '6'], answer: '4' },
        // Add more questions as needed
    ]);
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [selectedOption, setSelectedOption] = useState(null);
    const [score, setScore] = useState(0);

    const handleSubmit = () => {
        if (selectedOption === questions[currentQuestion].answer) {
            setScore(score + 1);
        }
        setCurrentQuestion(currentQuestion + 1);
        setSelectedOption(null);
    };

    return (
        <div className="quiz-game">
            <h2>Quiz Game</h2>
            {currentQuestion < questions.length ? (
                <div>
                    <p>{questions[currentQuestion].question}</p>
                    <div className="options">
                        {questions[currentQuestion].options.map((option, index) => (
                            <div
                                key={index}
                                className={`option ${selectedOption === option ? 'selected' : ''}`}
                                onClick={() => setSelectedOption(option)}
                            >
                                {option}
                            </div>
                        ))}
                    </div>
                    <button onClick={handleSubmit}>Submit</button>
                </div>
            ) : (
                <h3>Your score: {score}</h3>
            )}
        </div>
    );
};

export default QuizGame;
