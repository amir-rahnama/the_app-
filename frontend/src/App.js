import React, { useState } from 'react';

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

export default YourComponent;