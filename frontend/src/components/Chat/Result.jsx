import React from 'react';

const Result = ({ results = [] }) => {
  return (
    <div style={{ marginTop: 16 }}>
      <h4>Kết quả</h4>
      {results.length === 0 ? (
        <div style={{ color: '#888' }}>Chưa có kết quả nào</div>
      ) : (
        <ul style={{ paddingLeft: 16, color: '#444', fontSize: 14 }}>
          {results.map((result, idx) => (
            <li key={idx}>{result}</li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Result;
