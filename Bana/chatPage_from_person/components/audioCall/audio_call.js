import React, { useState } from 'react';
import './AudioCall.css';

const ChatBot = () => {
    const [userQuery, setUserQuery] = useState('');
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSend = async () => {
        if (!userQuery) {
            alert('Please enter a query');
            return;
        }

        setLoading(true);

        const body = JSON.stringify({
            user_query: userQuery,
        });

        try {
            const res = await fetch('https://visualize-abf4.onrender.com/search_image/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: body,
            });

            const data = await res.json();
            setResponse(data);
        } catch (error) {
            console.error('Error fetching data:', error);
            alert('Something went wrong. Please try again later.');
        }

        setLoading(false);
    };

    return (
        <div className="chat-container">
            <h1 className="cha-title">AI Image Search ChatBot</h1>

            <div className="chat-interface">
                <textarea
                    placeholder="Type your query..."
                    value={userQuery}
                    onChange={(e) => setUserQuery(e.target.value)}
                />
                <button onClick={handleSend} disabled={loading}>
                    {loading ? 'Searching...' : 'Send'}
                </button>
            </div>

            {response && (
                <div className="chat-response">
                    <h2>Response</h2>
                    <p><strong>Message:</strong> {response.message}</p>
                    <p><strong>Search Terms:</strong> {response.search_terms}</p>

                    <div className="image-gallery">
                        {response.image_metadata.map((image, index) => (
                            <div key={index} className="image-card">
                                <img src={image.url} alt={image.alt_text} />
                                <p>{image.alt_text}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ChatBot;
