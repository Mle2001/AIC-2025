import React from 'react';

function Keyframe({ keyframes, onSelect }) {
  // Sắp xếp theo độ chính xác giảm dần
  const sorted = [...keyframes].sort((a, b) => b.confidence - a.confidence);
  return (
    <div>
      {sorted.length === 0 && <p>Chưa có kết quả keyframe.</p>}
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {sorted.map((kf, idx) => (
          <li key={kf.id || idx} style={{ marginBottom: 12, cursor: 'pointer' }} onClick={() => onSelect(kf)}>
            <div style={{ background: '#f6f8fa', borderRadius: 6, padding: 12, boxShadow: '0 1px 4px #eee' }}>
              <span role="img" aria-label="keyframe">🖼️</span> {kf.label} <br />
              <span style={{ color: '#888' }}>{kf.time} - {Math.round(kf.confidence * 100)}% confidence</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Keyframe;
