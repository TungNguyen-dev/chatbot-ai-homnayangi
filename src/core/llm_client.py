"""
LLM client wrapper for OpenAI API (or other LLM providers).
"""
import chromadb
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Optional
from openai import OpenAI
from src.config.settings import settings
from src.core.calling_external_api import Get_weather


class LLMClient:
    """Wrapper around OpenAI API for LLM interactions."""

    def __init__(self):
        self.client = OpenAI(base_url=settings.OPENAI_BASE_URL, api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.OPENAI_MAX_TOKENS

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Whether to stream the response

        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream,
            )

            if stream:
                return response

            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return f"I apologize, but I encountered an error: {str(e)}"

    def generate_response_stream(
            self,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
    ):
        """
        Generate a streaming response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Yields:
            Response chunks as they are generated
        """
        try:
            function_definition = [{
                "type": "function",
                "function": {
                    "name": "how_to_cook_food",
                    "description": "Explain how to cook a specific food",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "food_name": {
                                "type": "string",
                                "description": "Food name extracted from user input"
                            },
                            "location": {
                                "type": "string",
                                "description": "Location extracted from user input"
                            }
                        },
                        "required": ["food_name"]
                    }
                }
            }, {
                "type": "function",
                "function": {
                    "name": "recommend_food",
                    "description": "Recommend food",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "gender": {
                                "type": "string",
                                "description": "Gender extracted from user input"
                            },
                            "location": {
                                "type": "string",
                                "description": "Location extracted from user input"
                            },
                            "disease ": {
                                "type": "string",
                                "description": "Disease name extracted from user input"
                            },
                            "time": {
                                "type": "string",
                                "description": "times of day extracted from user input"
                            }
                        },
                        "required": []
                    }
                }
            },

                {
                    "type": "function",
                    "function": {
                        "name": "find_restaurants",
                        "description": "Tìm kiếm các quán ăn, có thể lọc theo loại ẩm thực và địa điểm. Không cần hỏi thêm",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "Thành phố hoặc khu vực cần tìm kiếm nếu không có thì trả null",
                                },
                                "cuisine": {
                                    "type": "string",
                                    "description": "Loại ẩm thực cụ thể cần tìm, ví dụ: 'Phở', 'Pizza', 'Bún chả'",
                                },
                            },
                            # "required": ["location"],
                        },
                    }
                },
                # {
                #     "type": "function",
                #     "function": {
                #         "name": "get_current_weather",
                #         "description": "Lấy thông tin thời tiết hiện tại như nhiệt độ, tình trạng (mưa, nắng) tại một địa điểm.",
                #         "parameters": {
                #             "type": "object",
                #             "properties": {
                #                 "location": {
                #                     "type": "string",
                #                     "description": "Thành phố hoặc địa điểm, ví dụ: Hà Nội, Sài Gòn",
                #                 },
                #             },
                #             "required": ["location"],
                #         },
                #     }
                # },
                {
                    "type": "function",
                    "function": {
                        "name": "get_food_recommendation",
                        "description": "Gợi ý món ăn phù hợp dựa trên địa điểm và điều kiện thời tiết.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "Thành phố hoặc địa điểm, ví dụ: Hà Nội, Sài Gòn",
                                },
                                "weather_condition": {
                                    "type": "string",
                                    "description": "Mô tả nhiệt độ, ví dụ: '18 độ', '19 độ'",
                                },
                            },
                            "required": ["location", "weather_condition"],
                        },
                    }
                },
                {
                "type": "function",
                "function": {
                    "name": "recommend_food_detail",
                    "description": "Recommend food detail",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "style": {
                                "type": "string",
                                "description": "Style extracted from user input"
                            },
                            "taste": {
                                "type": "string",
                                "description": "Taste extracted from user input"
                            }
                        },
                        "required": []
                    }
                }
            }
            ]

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
                tools=function_definition,
                tool_choice="auto"
            )

            arguments_buffer = ""
            function_name = None

            for chunk in stream:
                delta = chunk.choices[0].delta

                if delta.content:
                    yield delta.content
                elif delta.tool_calls:
                    tool_call = delta.tool_calls[0]
                    function_call = tool_call.function

                    if function_call.name:
                        function_name = function_call.name
                    if function_call.arguments:
                        arguments_buffer += function_call.arguments

            # ✅ Sau khi stream kết thúc, parse arguments nếu có
            if function_name == "how_to_cook_food" and arguments_buffer:
                args = json.loads(arguments_buffer)
                food_name = args.get("food_name")
                location = args.get("location")
                print("Food name:", food_name)
                print("Location:", location)
                yield self.how_to_cook(food_name, location)
            elif function_name == "recommend_food_detail" and arguments_buffer:
                args = json.loads(arguments_buffer)
                style = args.get("style")
                taste = args.get("taste")
                print("style:", style)
                print("taste:", taste)
                yield self.recommend_food_detail(style, taste)
            elif function_name == "recommend_food" and arguments_buffer:
                args = json.loads(arguments_buffer)
                gender = args.get("gender")
                disease = args.get("disease")
                location = args.get("location")
                time = args.get("time")
                print("gender:", gender)
                print("disease:", disease)
                print("Location:", location)
                print("time:", time)
                yield self.recommend_food(disease, location, gender, time)
            elif function_name == "get_current_weather" and arguments_buffer:
                get_weather = Get_weather()
                try:
                    weather_and_location = get_weather.get_location_and_weather()
                    temperature = weather_and_location.get("temperature")
                    city = weather_and_location.get("city")
                    yield f"Thời tiết ở {city} hôm nay là {temperature} độ"
                except Exception as ex:
                    print (f"Error: {str(ex)}")
                    yield "Hiện tại có vấn đề, thử lại sau nhé"
            elif function_name == "get_food_recommendation" and arguments_buffer:
                args = json.loads(arguments_buffer)
                weather_temp = args.get("weather_condition")
                location = args.get("location")
                print("Location:", location)
                print("weather:", weather_temp)
                yield self.recommend_food_with_weather(weather_temp, location)
            elif function_name == "find_restaurants" and arguments_buffer:
                args = json.loads(arguments_buffer)
                location = args.get("location")
                # cuisine = args.get("cuisine")
                # print("Location1: " + location)
                print("Vao ham find_restaurants:")

                if not location or location == 'None':
                    get_weather = Get_weather()
                    weather_and_location = get_weather.get_location_and_weather()
                    city = weather_and_location.get("city")
                    print("Location: " + city)
                    yield self.find_restaurants(city, "")
                else:
                    yield self.find_restaurants(location, "")

        except Exception as e:
            yield f"Error: {str(e)}"

    def recommend_food(
            self,
            disease: str,
            location: str,
            time: str,
            gender: str
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user",
                       "content": f"Recommend me some dishes based on the following criteria: {disease},{location},{time},{gender}, using Vietnamese."}],
            temperature=0.2,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def recommend_food_detail(
        self,
        style: str,
        taste: str
    ) -> str:

        print("Bat dau xu ly fuction .....")
        # Dữ liệu mẫu
        dishes = [
            {"name": "Phở bò", "description": "Món nước, vị thanh, không cay"},
            {"name": "Bún chả", "description": "Món khô, vị nướng, không cay"},
            {"name": "Bún riêu cay", "description": "Món nước, vị cay nồng"},
            {"name": "Cơm tấm", "description": "Món khô, không cay"},
            {"name": "Bún đậu mắm tôm", "description": "Món khô, vị mắm, không cay"},
            {"name": "Bún cá cay", "description": "Món nước, vị cay"},
            {"name": "Bún riêu", "description": "Món nước, vị chua, không cay"},
            {"name": "Soup", "description": "Món Soup, and spicy (rất cay)"},
            {"name": "Soup", "description": "Món Soup, and Sweet (ngọt nhẹ)"},
        ]

        # Khởi tạo ChromaDB và mô hình embedding
        client = chromadb.Client()
        collection = client.create_collection("dishes")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Thêm món ăn vào database, bổ sung IDs
        for i, dish in enumerate(dishes):
            emb = model.encode(dish["description"])
            collection.add(
                ids=[f"dish_{i}"],
                documents=[dish["name"]],
                metadatas=[{"description": dish["description"]}],
                embeddings=[emb]
            )

        print("Bat dau tao query truy van .....")

        # Nhận truy vấn từ người dùng
        user_query = f"Today, I want to food with {style},{taste}"
        query_emb = model.encode(user_query)

        print("Bat dau tim kiem .....")
        # Tìm kiếm món ăn phù hợp nhất
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=2
        )
        print("Ket qua: " + results['metadatas'][0][0]['description'])
        return results['metadatas'][0][0]['description']

    def how_to_cook(
            self,
            food_name: str,
            location: str
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user",
                       "content": f"Briefly explain how {food_name} is prepared in {location}, using no more than 5 short lines in Vietnamese."}],
            temperature=0.2,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def recommend_food_with_weather(self, weather_temp, location):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user",
                       "content": f"Gợi ý món ăn ngon ở {location} với chuyển đổi nhiệt độ {weather_temp} độ C thành cảm giác và dựa vào cảm giác để recommend món ăn."}],
            temperature=0.2,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
    # anhvd14
    def find_restaurants(self, location: str, cuisine: Optional[str] = None) -> str:
        """
        Mô phỏng việc gọi API để tìm các quán ăn tại một địa điểm,
        có thể lọc theo loại ẩm thực.
        """
        print(f"--- ⚡️ Đang 'gọi API' để tìm quán ăn: find_restaurants(location='{location}', cuisine='{cuisine}') ---")

        # Dữ liệu giả lập (mock data)
        all_restaurants = {
            "hanoi": [
                {"name": "Phở Thìn Lò Đúc", "address": "13 Lò Đúc, Hai Bà Trưng", "rating": 4.5, "cuisine": "Phở"},
                {"name": "Bún chả Hương Liên (Obama)", "address": "24 Lê Văn Hưu, Hai Bà Trưng", "rating": 4.3,
                 "cuisine": "Bún chả"},
                {"name": "Pizza 4P's Tràng Tiền", "address": "43 Tràng Tiền, Hoàn Kiếm", "rating": 4.6,
                 "cuisine": "Pizza"},
                {"name": "Chả cá Thăng Long", "address": "21 Đường Thành, Hoàn Kiếm", "rating": 4.2,
                 "cuisine": "Chả cá"},
                {"name": "Taco ngon", "address": "15 Hàng Buồm, Hoàn Kiếm", "rating": 4.8, "cuisine": "Món Nhật"},
            ],
            "hà nội": [
                {"name": "Phở Thìn Lò Đúc", "address": "13 Lò Đúc, Hai Bà Trưng", "rating": 4.5, "cuisine": "Phở"},
                {"name": "Bún chả Hương Liên (Obama)", "address": "24 Lê Văn Hưu, Hai Bà Trưng", "rating": 4.3,
                 "cuisine": "Bún chả"},
                {"name": "Pizza 4P's Tràng Tiền", "address": "43 Tràng Tiền, Hoàn Kiếm", "rating": 4.6,
                 "cuisine": "Pizza"},
                {"name": "Chả cá Thăng Long", "address": "21 Đường Thành, Hoàn Kiếm", "rating": 4.2,
                 "cuisine": "Chả cá"},
                {"name": "Taco ngon", "address": "15 Hàng Buồm, Hoàn Kiếm", "rating": 4.8, "cuisine": "Món Nhật"},
            ],
            "sài gòn": [
                {"name": "Cục Gạch Quán", "address": "10 Đặng Tất, Quận 1", "rating": 4.4, "cuisine": "Món Việt"},
                {"name": "Bánh mì Huỳnh Hoa", "address": "26 Lê Thị Riêng, Quận 1", "rating": 4.6,
                 "cuisine": "Bánh mì"},
            ],
            "thanh xuân": [
                {"name": "Cục Gạch Quán", "address": "10 Đặng Tất, Quận 1", "rating": 4.4, "cuisine": "Món Việt"},
                {"name": "Bánh mì Huỳnh Hoa", "address": "26 Lê Thị Riêng, Quận 1", "rating": 4.6,
                 "cuisine": "Bánh mì"}
            ]
        }

        # Tìm kiếm dựa trên địa điểm
        results = all_restaurants.get(location.lower(), [])

        # Lọc theo ẩm thực nếu có
        if cuisine:
            # Tìm kiếm linh hoạt (ví dụ: 'phở' sẽ khớp với 'Phở')
            results = [res for res in results if cuisine.lower() in res["cuisine"].lower()]
        print(results)
        return self.format_filtered_restaurants_to_text(results, location)

    def format_filtered_restaurants_to_text(self, filtered_list, location=""):
        """
        Chuyển đổi một danh sách nhà hàng đã được lọc thành chuỗi text.
        """
        if not filtered_list:
            return f"Rất tiếc, tôi không tìm thấy quán ăn nào phù hợp ở {location}."

        location_text = f" tại {location.title()}" if location else ""
        result_lines = [f"Đây là một vài gợi ý tuyệt vời{location_text} cho bạn:"]

        for restaurant in filtered_list:
            name = restaurant.get("name", "N/A")
            address = restaurant.get("address", "N/A")
            rating = restaurant.get("rating", "N/A")
            cuisine = restaurant.get("cuisine", "N/A")

            result_lines.append(f"\n✨ **{name}**")  # Dùng Markdown để làm nổi bật
            result_lines.append(f"   - **Địa chỉ:** {address}")
            result_lines.append(f"   - **Món ăn:** {cuisine} (Đánh giá: {rating} ⭐)")

        return "\n".join(result_lines)
