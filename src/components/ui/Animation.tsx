'use client';

import { useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';

const TARGET_TEXT = 'NoCode AIModel Builder';
// const TARGET_LINK = '/register';
const TARGET_LINK = '#';
const CYCLES_PER_LETTER = 3;
const SHUFFLE_TIME = 60;

const CHARS = '!@#$%^&*():{};|,.<>/?';

export const RegisterButton = () => {
  const router = useRouter();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [text, setText] = useState(TARGET_TEXT);

  const scramble = () => {
    let pos = 0;

    intervalRef.current = setInterval(() => {
      const scrambled = TARGET_TEXT.split('')
        .map((char, index) => {
          if (pos / CYCLES_PER_LETTER > index) {
            return char;
          }

          const randomCharIndex = Math.floor(Math.random() * CHARS.length);
          const randomChar = CHARS[randomCharIndex];

          return randomChar;
        })
        .join('');

      setText(scrambled);
      pos++;

      if (pos >= TARGET_TEXT.length * CYCLES_PER_LETTER) {
        stopScramble();
      }
    }, SHUFFLE_TIME);
  };

  const stopScramble = () => {
    clearInterval(intervalRef.current ?? undefined);

    setText(TARGET_TEXT);
  };

  return (
    <motion.button
      whileHover={{
        scale: 1.025,
      }}
      whileTap={{
        scale: 0.975,
      }}
      onMouseEnter={scramble}
      onMouseLeave={stopScramble}
      onClick={() => router.push(TARGET_LINK)}
      className='group relative hidden overflow-hidden rounded-full border-[1px]  px-4 py-2 font-mono font-medium uppercase text-neutral-300 transition-colors hover:text-indigo-300 md:block'
    >
      <div className='relative z-10 flex items-center gap-2 text-xl'>
        <span>{text}</span>
      </div>
    </motion.button>
  );
};