// app/page.tsx
import ProfileForm from "@/components/forms/Chatform";
import { RegisterButton } from "@/components/ui/Animation";
import Flow from "@/components/pages/Flow";
export default function Home() {
  return (
    <>
    
          <ProfileForm />
         <Flow />
    </>
  );
}