import React, { useRef, useEffect } from 'react';

function VideoPlayer({ videoUrl, timestamp }) {
  const videoRef = useRef();

  useEffect(() => {
    if (videoRef.current && timestamp) {
      videoRef.current.currentTime = timestamp;
    }
  }, [timestamp]);

  if (!videoUrl) {
    return <div style={{ textAlign: 'center', color: '#fff', background: '#222', padding: 40, borderRadius: 8 }}>Video Player</div>;
  }

  return (
    <div>
      <video ref={videoRef} src={videoUrl} controls style={{ width: '100%', borderRadius: 8, background: '#222' }} />
    </div>
  );
}

export default VideoPlayer;
