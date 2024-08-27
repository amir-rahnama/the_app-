/*import React, { useState } from 'react';

function YourComponent() {
    const [response, setResponse] = useState(null);

    const handleSubmit = async () => {
        const data = { key: 'value' }; // Replace with your actual data

        const res = await fetch('http://localhost:8000/api/data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await res.json();
        setResponse(result);
    };

    return (
        <div>
            <button onClick={handleSubmit}>Send Data</button>
            {response && <div>Response: {JSON.stringify(response)}</div>}
        </div>
    );
}

export default YourComponent;*/


import React, { useState } from 'react';

function YourComponent() {
    const [file, setFile] = useState(null);
    const [response, setResponse] = useState(null);
    const [error, setError] = useState('');

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && selectedFile.type.startsWith('audio/')) {
            setFile(selectedFile);
            setError('');
        } else {
            setError('Please upload a valid audio file.');
            setFile(null);
        }
    };

    const handleSubmit = async () => {
        if (!file) {
            setError('No audio file selected.');
            return;
        }

        const formData = new FormData();
        formData.append('audio', file);

        try {
            const res = await fetch('http://localhost:8000/api/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await res.json();
            setResponse(result);
        } catch (err) {
            console.error('Error uploading file:', err);
            setError('Failed to upload file.');
        }
    };

    return (
        <div>
            <input type="file" accept="audio/*" onChange={handleFileChange} />
            <button onClick={handleSubmit}>Upload Audio</button>
            {error && <div style={{ color: 'red' }}>{error}</div>}
            {response && <div>Response: {JSON.stringify(response)}</div>}
        </div>
    );
}

export default YourComponent;