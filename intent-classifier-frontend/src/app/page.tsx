'use client'

import { useState, useEffect } from 'react'

interface PredictionResult {
  intent: string
  confidence: number
  language: string
}

interface WeatherData {
  temp: number;
  description: string;
  icon: string;
  feelslike: number;
  humidity: number;
}

export default function Home() {
  const [mounted, setMounted] = useState(false)
  const [input, setInput] = useState('')
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [globalTimer, setGlobalTimer] = useState<number | null>(null)
  const [globalTimerRunning, setGlobalTimerRunning] = useState(false)
  const [weather, setWeather] = useState<WeatherData | null>(null)
  const [showDate, setShowDate] = useState(false)
  const [detectedLanguage, setDetectedLanguage] = useState<string>('en')

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    let intervalId: NodeJS.Timeout

    if (globalTimerRunning && globalTimer !== null) {
      intervalId = setInterval(() => {
        setGlobalTimer(prev => {
          if (prev === null || prev <= 0) {
            setGlobalTimerRunning(false)
            return null
          }
          return prev - 1
        })
      }, 1000)
    }

    return () => {
      if (intervalId) clearInterval(intervalId)
    }
  }, [globalTimerRunning, globalTimer])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const startListening = () => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition
      const recognition = new SpeechRecognition()
      
      recognition.continuous = false
      recognition.interimResults = false

      recognition.onstart = () => {
        setIsListening(true)
      }

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = event.results[0][0].transcript
        setInput(transcript)
      }

      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error('Speech recognition error:', event.error)
        setIsListening(false)
      }

      recognition.onend = () => {
        setIsListening(false)
      }

      recognition.start()
    } else {
      alert('Speech recognition is not supported in this browser.')
    }
  }

  const extractTimeFromText = (text: string): number | null => {
    const numbers = text.match(/\d+/g)
    if (!numbers) return null
    
    const timeValue = parseInt(numbers[0])
    if (text.includes('minute') || text.includes('minutes')) {
      return timeValue * 60
    } else if (text.includes('second') || text.includes('seconds')) {
      return timeValue
    }
    return timeValue * 60 // default to minutes
  }

  const fetchWeather = async (lat: number, lon: number) => {
    const API_KEY = 'ba750d11b83646bd94344746251601'
    try {
      const response = await fetch(
        `https://api.weatherapi.com/v1/current.json?key=${API_KEY}&q=${lat},${lon}&aqi=no`
      )
      
      if (!response.ok) {
        throw new Error(`Weather API error: ${response.status}`)
      }

      const data = await response.json()
      
      // Validate the response data structure
      if (!data.current) {
        throw new Error('Invalid weather data format')
      }
      
      return {
        temp: Math.round(data.current.temp_c),
        description: data.current.condition.text,
        icon: data.current.condition.icon,
        feelslike: Math.round(data.current.feelslike_c),
        humidity: data.current.humidity
      }
    } catch (error) {
      console.error('Error fetching weather:', error)
      alert('Unable to fetch weather data. Please try again later.')
      return null
    }
  }

  const formatDate = () => {
    const now = new Date()
    const options: Intl.DateTimeFormatOptions = { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }
    return now.toLocaleDateString('en-US', options)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    setIsLoading(true)
    try {
      const response = await fetch('http://localhost:5001/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text: input,
          language: detectedLanguage 
        }),
      })

      if (!response.ok) {
        throw new Error(`Failed to classify (${response.status})`)
      }

      const data = await response.json()
      setPrediction(data)
      setDetectedLanguage(data.language)

      if (data.intent === 'set_timer') {
        const seconds = extractTimeFromText(input)
        if (seconds) {
          setGlobalTimer(seconds)
          setGlobalTimerRunning(true)
        }
      } else if (data.intent === 'get_weather') {
        if ('geolocation' in navigator) {
          navigator.geolocation.getCurrentPosition(async (position) => {
            const weatherData = await fetchWeather(
              position.coords.latitude,
              position.coords.longitude
            )
            if (weatherData) {
              setWeather(weatherData)
              // Auto-hide weather after 30 seconds
              setTimeout(() => setWeather(null), 30000)
            }
          }, (error) => {
            console.error('Error getting location:', error)
            alert('Unable to get your location. Please enable location services.')
          })
        } else {
          alert('Geolocation is not supported by your browser')
        }
      } else if (data.intent === 'get_date') {
        setShowDate(true)
        // Auto-hide date after 30 seconds
        setTimeout(() => setShowDate(false), 30000)
      }
    } catch (error) {
      console.error('Classification error:', error)
      alert('Failed to connect to the server. Please make sure the backend is running.')
    } finally {
      setIsLoading(false)
    }
  }

  if (!mounted) return null

  return (
    <main className="min-h-screen bg-[#f5f5f7] dark:bg-black flex flex-col items-center justify-center p-4">
      <div className="fixed top-8 flex gap-4">
        {globalTimer !== null && (
          <div className="bg-white dark:bg-[#1d1d1f] rounded-2xl shadow-lg p-6 flex flex-col items-center space-y-2
                       border border-gray-200 dark:border-gray-800">
            <div className="text-4xl font-mono font-semibold text-gray-900 dark:text-white">
              {Math.floor(globalTimer / 60).toString().padStart(2, '0')}:
              {(globalTimer % 60).toString().padStart(2, '0')}
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${globalTimerRunning ? 'bg-blue-500 animate-pulse' : 'bg-gray-300 dark:bg-gray-600'}`} />
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {globalTimerRunning ? 'Timer Running' : 'Timer Paused'}
              </span>
            </div>
            <button
              onClick={() => setGlobalTimer(null)}
              className="text-sm text-red-500 hover:text-red-600 dark:text-red-400 dark:hover:text-red-300"
            >
              Cancel Timer
            </button>
          </div>
        )}
        
        {weather && (
          <div className="bg-white dark:bg-[#1d1d1f] rounded-2xl shadow-lg p-6 flex flex-col items-center space-y-3
                 border border-gray-200 dark:border-gray-800">
            <div className="text-4xl font-semibold text-gray-900 dark:text-white flex items-center gap-2">
              {weather.temp}°C
              <img 
                src={weather.icon}
                alt={weather.description}
                className="w-12 h-12"
              />
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400 capitalize">
              {weather.description}
            </div>
            <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
              <div>
                Feels like: {weather.feelslike}°C
              </div>
              <div>
                Humidity: {weather.humidity}%
              </div>
            </div>
          </div>
        )}
        
        {showDate && (
          <div className="bg-white dark:bg-[#1d1d1f] rounded-2xl shadow-lg p-6 flex flex-col items-center space-y-3
                 border border-gray-200 dark:border-gray-800">
            <div className="text-2xl font-semibold text-gray-900 dark:text-white">
              {formatDate()}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Current Date & Time
            </div>
          </div>
        )}
      </div>
      <div className="w-full max-w-md bg-white dark:bg-[#1d1d1f] rounded-2xl shadow-lg p-8 space-y-8">
        <h1 className="text-2xl font-semibold text-center text-gray-900 dark:text-white">
          Intent Classifier
        </h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="What would you like to do?"
              className="w-full h-11 px-4 rounded-lg bg-gray-50 dark:bg-[#2c2c2e] border border-gray-200 dark:border-gray-700 
                       focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400
                       text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400
                       transition-all duration-200"
              disabled={isLoading}
            />
            <button
              type="button"
              onClick={startListening}
              className="mx-auto flex items-center justify-center w-12 h-12 rounded-full 
                       bg-gray-100 dark:bg-[#2c2c2e] text-gray-500 dark:text-gray-400
                       hover:text-blue-500 dark:hover:text-blue-400 
                       hover:bg-gray-200 dark:hover:bg-[#3c3c3e]
                       transition-all duration-200"
            >
              {isListening ? (
                <div className="relative w-6 h-6">
                  <div className="absolute inset-0 animate-ping rounded-full bg-blue-500 opacity-75"></div>
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z" />
                  </svg>
                </div>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z" />
                </svg>
              )}
            </button>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-500 hover:bg-blue-600 active:bg-blue-700
                     text-white font-medium py-3 px-4 rounded-lg
                     transition-all duration-200 transform active:scale-[0.98]
                     disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <div className="flex items-center justify-center space-x-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Processing...</span>
              </div>
            ) : (
              'Classify Intent'
            )}
          </button>
        </form>

        {prediction && (
          <div className="mt-8 p-6 bg-gray-50 dark:bg-[#2c2c2e] rounded-xl space-y-4 
                         border border-gray-100 dark:border-gray-800">
            <div className="space-y-3">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Detected Intent</p>
                <p className="text-lg font-medium text-gray-900 dark:text-white">
                  {prediction.intent.replace(/_/g, ' ').toLowerCase()}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Confidence</p>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-500 rounded-full transition-all duration-500"
                      style={{ width: `${prediction.confidence}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {prediction.confidence.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="text-sm text-gray-500 dark:text-gray-400">
          Detected Language: {detectedLanguage.toUpperCase()}
        </div>
      </div>
    </main>
  )
} 